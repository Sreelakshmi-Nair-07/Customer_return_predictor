import os
import json
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, precision_score
from sklearn.calibration import calibration_curve

import joblib


def ensure_dirs(base="reports"):
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    figs = p / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    return p, figs


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return None


def summary_stats(df):
    stats = {}
    stats["shape"] = df.shape
    stats["missing_count"] = df.isna().sum().to_dict()
    stats["dtypes"] = df.dtypes.astype(str).to_dict()
    stats["numeric_summary"] = df.select_dtypes(include=[np.number]).describe().to_dict()
    return stats


def _coerce_target_to_numeric(df, target_col):
    """Try to make the target column numeric (0/1). Returns a Series or None if not possible."""
    if target_col not in df.columns:
        return None
    ser = df[target_col]
    if pd.api.types.is_numeric_dtype(ser):
        return ser
    # common string labels
    mapping = {"Returned": 1, "Not Returned": 0, "returned": 1, "not returned": 0, "Yes": 1, "No": 0}
    if ser.dropna().isin(mapping.keys()).all():
        return ser.map(mapping)
    # try to map boolean-like
    if ser.dropna().isin([0, 1, "0", "1"]).all():
        return pd.to_numeric(ser, errors='coerce')
    # last resort: factorize and if only two levels, map largest->1
    labels, uniques = pd.factorize(ser.fillna("__nan__"))
    uniq = pd.Series(uniques)
    if len(uniq) == 2:
        return pd.Series(labels).replace({0: 0, 1: 1})
    return None


def plot_target_distribution(df, target_col, figs):
    if target_col not in df.columns:
        print(f"Target column {target_col} not in dataframe")
        return
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=df)
    plt.title("Target distribution")
    plt.tight_layout()
    out = figs / "target_distribution.png"
    plt.savefig(out)
    plt.close()
    print("Saved:", out)


def plot_numeric_corr(df, figs, max_features=40):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return
    # limit columns to avoid huge heatmaps
    cols = num.columns[:max_features]
    corr = num[cols].corr()
    plt.figure(figsize=(min(12, len(cols)), min(10, len(cols))))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Numeric feature correlation")
    plt.tight_layout()
    out = figs / "numeric_correlation.png"
    plt.savefig(out)
    plt.close()
    print("Saved:", out)






def plot_seasonality(df, date_col_candidates, target_col, figs):
    for dc in date_col_candidates:
        if dc in df.columns:
            try:
                d = pd.to_datetime(df[dc], errors='coerce')
                df['_month'] = d.dt.month
                ynum = _coerce_target_to_numeric(df, target_col)
                if ynum is None:
                    print(f"Skipping seasonality for {dc}: cannot coerce target '{target_col}' to numeric in this dataframe")
                    continue
                grp = pd.DataFrame({'_month': df['_month'], '_y': ynum}).groupby('_month')['_y'].mean()
                plt.figure(figsize=(8, 4))
                sns.lineplot(x=grp.index, y=grp.values)
                plt.xticks(range(1, 13))
                plt.xlabel('Month')
                plt.ylabel(f"{target_col} rate")
                plt.title(f"Monthly return rate (from {dc})")
                plt.tight_layout()
                out = figs / f"monthly_return_rate_from_{dc}.png"
                plt.savefig(out)
                plt.close()
                print("Saved:", out)
            except Exception as e:
                print("Seasonality plot failed for column", dc, e)


def plot_precision_recall_curve(y, probs, figs):
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y, probs)
        ap = average_precision_score(y, probs)
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, label=f'AP = {ap:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        out = figs / 'precision_recall_curve.png'
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print('Saved:', out)
    except Exception as e:
        print('Precision-Recall plot failed:', e)



def plot_time_trends(df, date_col_candidates, target_col, figs):
    # hourly, weekday, monthly trends
    for dc in date_col_candidates:
        if dc in df.columns:
            try:
                d = pd.to_datetime(df[dc], errors='coerce')
                ynum = _coerce_target_to_numeric(df, target_col)
                if ynum is None:
                    print(f"Skipping time trends for {dc}: cannot coerce target '{target_col}' to numeric in this dataframe")
                    continue
                tmp = pd.DataFrame({'dt': d, '_y': ynum}).dropna()
                if tmp.empty:
                    continue
                tmp['hour'] = tmp['dt'].dt.hour
                tmp['weekday'] = tmp['dt'].dt.day_name()
                tmp['month'] = tmp['dt'].dt.month

                # hourly
                grp_h = tmp.groupby('hour')['_y'].mean()
                plt.figure(figsize=(10, 4))
                sns.lineplot(x=grp_h.index, y=grp_h.values)
                plt.xlabel('Hour of day')
                plt.ylabel('Return rate')
                plt.title(f'Hourly return rate (from {dc})')
                plt.tight_layout()
                out = figs / f'hourly_return_rate_from_{dc}.png'
                plt.savefig(out)
                plt.close()
                print('Saved:', out)

                # weekday
                grp_w = tmp.groupby('weekday')['_y'].mean().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
                plt.figure(figsize=(10, 4))
                sns.barplot(x=grp_w.index, y=grp_w.values)
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('Return rate')
                plt.title(f'Weekday return rate (from {dc})')
                plt.tight_layout()
                out = figs / f'weekday_return_rate_from_{dc}.png'
                plt.savefig(out)
                plt.close()
                print('Saved:', out)

                # monthly (reuse previous)
                grp_m = tmp.groupby('month')['_y'].mean()
                plt.figure(figsize=(8, 4))
                sns.lineplot(x=grp_m.index, y=grp_m.values)
                plt.xticks(range(1,13))
                plt.xlabel('Month')
                plt.ylabel('Return rate')
                plt.title(f'Monthly return rate (from {dc})')
                plt.tight_layout()
                out = figs / f'monthly_return_rate_from_{dc}.png'
                plt.savefig(out)
                plt.close()
                print('Saved:', out)
            except Exception as e:
                print('Time trend failed for', dc, e)


def plot_risk_score_pie(probs, figs):
    """Create pie chart of risk score distribution in three buckets: Low/Medium/High"""
    try:
        # Use three buckets: Low / Medium / High
        # Default thresholds: <=0.33 low, 0.33-0.66 medium, >0.66 high
        t1, t2 = 0.33, 0.66
        bins = [0.0, t1, t2, 1.0]
        labels = ['Low', 'Medium', 'High']
        cat = pd.cut(probs, bins=bins, labels=labels, include_lowest=True)
        counts = cat.value_counts().reindex(labels, fill_value=0)

        plt.figure(figsize=(8, 6))
        plt.pie(counts.values, labels=[f"{lab} ({int(c)})" for lab, c in counts.items()], autopct='%1.1f%%',
                colors=['lightgreen', 'gold', 'salmon'])
        plt.title('Distribution of Return Risk Scores (Low / Medium / High)')
        out = figs / 'risk_score_pie.png'
        plt.savefig(out)
        plt.close()
        print('Saved:', out)
    except Exception as e:
        print('Risk score pie chart failed:', e)



def plot_roc_auc(y, probs, figs):
    try:
        auc = roc_auc_score(y, probs)
        fpr, tpr, _ = roc_curve(y, probs)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend()
        out = figs / 'roc_curve.png'
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print('Saved:', out)
    except Exception as e:
        print('ROC plot failed:', e)

def plot_learning_curve(model, X, y, figs):
    """Plot learning curve to detect under/overfitting"""
    try:
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='roc_auc'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.plot(train_sizes, val_mean, label='Cross-validation score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        
        plt.xlabel('Training Examples')
        plt.ylabel('ROC AUC Score')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid()
        
        out = figs / 'learning_curve.png'
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print('Saved:', out)
    except Exception as e:
        print('Learning curve plot failed:', e)

def plot_metrics_by_segment(df, probs, y_test, segment_cols, figs):
    """Plot metrics breakdown by different segments (Category, Country, SalesChannel)"""
    for col in segment_cols:
        if col not in df.columns:
            continue
            
        try:
            # Create DataFrame with predictions and actual values
            results = pd.DataFrame({
                'segment': df[col],
                'prob': probs,
                'actual': y_test
            })
            
            # Calculate metrics per segment
            metrics = []
            for segment in results['segment'].unique():
                mask = results['segment'] == segment
                if mask.sum() < 10:  # Skip segments with too few samples
                    continue
                    
                segment_probs = results.loc[mask, 'prob']
                segment_actual = results.loc[mask, 'actual']
                
                try:
                    auc = roc_auc_score(segment_actual, segment_probs)
                    avg_risk = segment_probs.mean()
                    return_rate = segment_actual.mean()
                    count = len(segment_probs)
                    
                    metrics.append({
                        'segment': segment,
                        'AUC': auc,
                        'Avg Risk Score': avg_risk,
                        'Return Rate': return_rate,
                        'Count': count
                    })
                except Exception:
                    continue
            
            if not metrics:
                continue
                
            # Create visualization
            metrics_df = pd.DataFrame(metrics)
            metrics_df = metrics_df.sort_values('Count', ascending=False).head(10)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Return Rate and Risk Score
            metrics_df.plot(x='segment', y=['Return Rate', 'Avg Risk Score'], 
                          kind='bar', ax=ax1)
            ax1.set_title(f'Return Rate and Risk Score by {col}')
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: AUC Score
            sns.barplot(data=metrics_df, x='segment', y='AUC', ax=ax2)
            ax2.set_title(f'AUC Score by {col}')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            out = figs / f'metrics_by_{col}.png'
            plt.savefig(out)
            plt.close()
            print('Saved:', out)
            
        except Exception as e:
            print(f'Metrics by {col} plot failed:', e)


def plot_sales_channel_reliability(df, probs, y_test, channel_col, figs):
    """Create sales channel reliability index visualization"""
    if channel_col not in df.columns:
        return
        
    try:
        # Calculate metrics per channel
        results = pd.DataFrame({
            'channel': df[channel_col],
            'prob': probs,
            'actual': y_test
        })
        
        metrics = []
        for channel in results['channel'].unique():
            mask = results['channel'] == channel
            if mask.sum() < 10:  # Skip channels with too few samples
                continue
                
            channel_probs = results.loc[mask, 'prob']
            channel_actual = results.loc[mask, 'actual']
            
            # Calculate return rate and precision
            return_rate = channel_actual.mean()
            precision = precision_score(channel_actual, channel_probs > 0.5)
            
            # Reliability index = (1 - return_rate) * precision
            reliability = (1 - return_rate) * precision
            
            metrics.append({
                'channel': channel,
                'return_rate': return_rate,
                'precision': precision,
                'reliability': reliability,
                'count': len(channel_probs)
            })
        
        if not metrics:
            return
            
        metrics_df = pd.DataFrame(metrics)
        metrics_df = metrics_df.sort_values('count', ascending=False)
        
        plt.figure(figsize=(12, 6))
        
        # Plot stacked bars of return rate and precision
        bottom = np.zeros(len(metrics_df))
        
        plt.bar(metrics_df['channel'], metrics_df['return_rate'], 
                label='Return Rate', alpha=0.6)
        plt.bar(metrics_df['channel'], metrics_df['precision'],
                bottom=metrics_df['return_rate'], label='Precision', alpha=0.6)
        
        # Add reliability score as text
        for i, row in metrics_df.iterrows():
            plt.text(i, 1.05, f'Reliability: {row["reliability"]:.2f}',
                    ha='center', va='bottom')
        
        plt.xlabel('Sales Channel')
        plt.ylabel('Score')
        plt.title('Sales Channel Reliability Analysis')
        plt.legend()
        plt.xticks(rotation=45)
        
        out = figs / 'sales_channel_reliability.png'
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print('Saved:', out)
        
    except Exception as e:
        print('Sales channel reliability plot failed:', e)

def generate_risk_report(raw_df, X_for_pred, probs, y_test, figs, report_dir, top_n=20):
    # Create a DataFrame with risk scores and, if possible, attach raw metadata (CustomerID, Category, Country)
    recs = None
    try:
        if raw_df is not None and len(raw_df) >= len(probs):
            base = raw_df.reset_index(drop=True).iloc[:len(probs)].copy()
            base['risk_score'] = probs
            recs = base
        else:
            # fallback: use X_for_pred columns if they contain IDs
            base = X_for_pred.reset_index(drop=True).iloc[:len(probs)].copy()
            base['risk_score'] = probs
            recs = base
    except Exception as e:
        print('Could not attach metadata to risk scores:', e)
        recs = pd.DataFrame({'risk_score': probs})

    # Save top N risky
    recs_sorted = recs.sort_values('risk_score', ascending=False)
    out_csv = Path(report_dir) / 'risk_scores.csv'
    recs.to_csv(out_csv, index=False)
    out_top = Path(report_dir) / 'risk_top_{}.csv'.format(top_n)
    recs_sorted.head(top_n).to_csv(out_top, index=False)
    print('Saved risk scores to', out_csv, 'and top to', out_top)

    # Insights: group by category/country/discount bins
    insights = {}
    for col in ['Category', 'Country', 'Discount', 'PaymentMethod']:
        if col in recs.columns:
            try:
                if col == 'Discount':
                    series = pd.to_numeric(recs[col], errors='coerce')
                    recs['discount_bin'] = pd.qcut(series.fillna(0), q=5, duplicates='drop')
                    grp = recs.groupby('discount_bin')['risk_score'].agg(['mean', 'count']).to_dict()
                else:
                    grp = recs.groupby(col)['risk_score'].agg(['mean', 'count']).sort_values('count', ascending=False).head(20).to_dict()
                insights[col] = grp
            except Exception as e:
                print('Insight grouping failed for', col, e)

    # Save insights JSON
    out_ins = Path(report_dir) / 'risk_insights.json'
    with open(out_ins, 'w') as f:
        json.dump(insights, f, default=str, indent=2)
    print('Saved insights to', out_ins)

    # Plot top categories by average risk if available
    if 'Category' in recs.columns:
        try:
            grp = recs.groupby('Category')['risk_score'].mean().sort_values(ascending=False).head(20)
            plt.figure(figsize=(10, 5))
            sns.barplot(x=grp.index.astype(str), y=grp.values)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Average risk score')
            plt.title('Top categories by average return risk')
            out = Path(report_dir) / 'top_categories_by_risk.png'
            plt.tight_layout()
            plt.savefig(out)
            plt.close()
            print('Saved:', out)
        except Exception as e:
            print('Failed plotting top categories by risk', e)



def risk_score_visualizations(X, y, preprocessor_path, model_path, figs):
    # Try to load preprocessor and model; if not available assume X is already numeric
    model = None
    pre = None
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print("Loaded model from", model_path, "class:", type(model).__name__)
        except Exception as e:
            print("Could not load model:", e)
    if os.path.exists(preprocessor_path):
        try:
            pre = joblib.load(preprocessor_path)
            print("Loaded preprocessor from", preprocessor_path)
        except Exception as e:
            print("Could not load preprocessor:", e)

    if model is None:
        print("No trained model available — skipping risk score visualizations")
        return

    X_proc = X
    if pre is not None:
        try:
            X_proc = pre.transform(X)
        except Exception as e:
            print("Preprocessor transform failed, assuming X is ready:", e)

    # predicted probabilities
    try:
        probs = model.predict_proba(X_proc)[:, 1]
    except Exception:
        # maybe estimator exposes decision_function
        try:
            probs = model.decision_function(X_proc)
            probs = (probs - probs.min()) / (probs.max() - probs.min())
        except Exception as e:
            print("Could not obtain probabilities or scores:", e)
            return

    # histogram of probabilities (risk scores)
    plt.figure(figsize=(8, 4))
    sns.histplot(probs, bins=50, kde=True)
    plt.title('Predicted return risk score distribution')
    plt.xlabel('Predicted probability')
    out = figs / 'risk_score_histogram.png'
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print('Saved:', out)

    # Return probabilities for downstream reports
    # If y is available, plot ROC and calibration
    if y is None:
        print('No y provided for ROC/calibration plots; skipping those.')
    elif len(y) != len(probs):
        print(f'y length ({len(y)}) does not match predicted probs length ({len(probs)}); skipping ROC/calibration.')
    else:
        try:
            auc = roc_auc_score(y, probs)
            fpr, tpr, _ = roc_curve(y, probs)
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], '--', color='gray')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC Curve')
            plt.legend()
            out = figs / 'roc_curve.png'
            plt.tight_layout()
            plt.savefig(out)
            plt.close()
            print('Saved:', out)
        except Exception as e:
            print('ROC plot failed:', e)

        try:
            prob_true, prob_pred = calibration_curve(y, probs, n_bins=10)
            plt.figure(figsize=(6, 6))
            plt.plot(prob_pred, prob_true, marker='o')
            plt.plot([0, 1], [0, 1], '--', color='gray')
            plt.xlabel('Mean predicted probability')
            plt.ylabel('Fraction of positives')
            plt.title('Calibration curve')
            out = figs / 'calibration_curve.png'
            plt.tight_layout()
            plt.savefig(out)
            plt.close()
            print('Saved:', out)
        except Exception as e:
            print('Calibration plot failed:', e)
    return probs


def main():
    report_dir, figs = ensure_dirs()
    params = load_params()

    # Try processed data first, otherwise raw
    processed_train = params.get('paths', {}).get('processed_train', 'data/processed/train.csv')
    processed_test = params.get('paths', {}).get('processed_test', 'data/processed/test.csv')
    raw = params.get('paths', {}).get('raw', params.get('dataset', {}).get('raw_csv_path'))
    target_col = params.get('dataset', {}).get('target_col', 'target')

    df_train = safe_read_csv(processed_train)
    if df_train is None:
        df_train = safe_read_csv(raw)
    df_test = safe_read_csv(processed_test)

    if df_train is None:
        print('No data available for EDA. Check paths in params.yaml')
        return

    # Build a combined sample for EDA (train + test small)
    df_sample = df_train.copy()
    if df_test is not None:
        df_sample = pd.concat([df_sample, df_test.sample(min(len(df_test), 1000))], ignore_index=True)

    # Save a high-level summary
    summary = summary_stats(df_sample)
    with open(report_dir / 'eda_summary.json', 'w') as f:
        json.dump(summary, f, default=str, indent=2)
    print('Saved EDA summary to', report_dir / 'eda_summary.json')

    # We'll produce only the focused plots requested:
    # - Risk score pie & quartiles
    # - Learning curve
    # - Metrics by Category/Country/SalesChannel
    # - Customer segment (quartiles)
    # - Discount sensitivity curve
    # - ROC AUC and Precision-Recall
    # - Monthly, hourly, weekly return rates

    # Load raw data and choose seasonality columns
    raw_df = safe_read_csv(raw)
    season_df = raw_df if raw_df is not None else df_sample
    date_candidates = ['InvoiceDate', 'Invoice_Date', 'Invoice Date', 'OrderDate', 'Order_Date', 'Date', 'TransactionDate', 'OrderPlacedDate', 'order_date']

    # Load y_test explicitly if available
    y_test = None
    y_test_path = Path('data/processed/y_test.csv')
    if y_test_path.exists():
        try:
            y_test = pd.read_csv(y_test_path).values.ravel()
        except Exception as e:
            print('Could not read data/processed/y_test.csv:', e)
    else:
        if df_test is not None and target_col in df_test.columns:
            y_test = df_test[target_col].values

    preprocessor_path = params.get('paths', {}).get('preprocessor', 'models/preprocessor.pkl')
    # Prefer registry if available
    registry_path = Path('models/models_registry.json')
    model_path = params.get('paths', {}).get('model', 'models/model.pkl')
    if registry_path.exists():
        try:
            reg = json.loads(registry_path.read_text())
            preferred = params.get('model', {}).get('type')
            if preferred and preferred in reg:
                model_path = reg[preferred]
            elif 'best' in reg:
                model_path = reg['best']
            else:
                model_path = next(iter(reg.values()))
        except Exception as e:
            print('Could not read models_registry.json:', e)

    # Try use df_test features for prediction; if processed_test exists, use it
    if safe_read_csv(processed_test) is not None:
        X_for_pred = pd.read_csv(processed_test)
    else:
        # drop target if present
        X_for_pred = df_test.copy() if df_test is not None else df_train.copy()
        if target_col in X_for_pred.columns:
            X_for_pred = X_for_pred.drop(columns=[target_col])

    # Load model and preprocessor if available
    model = None
    preprocessor = None
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print('Loaded model', type(model).__name__)
        except Exception as e:
            print('Could not load model for focused EDA:', e)
    if os.path.exists(preprocessor_path):
        try:
            preprocessor = joblib.load(preprocessor_path)
            print('Loaded preprocessor')
        except Exception as e:
            print('Could not load preprocessor for focused EDA:', e)

    # Compute probabilities if model available
    probs = None
    if model is not None:
        try:
            # Use X_for_pred if available
            X_pred = X_for_pred.reset_index(drop=True).iloc[:len(df_sample)] if 'X_for_pred' in locals() else df_sample
            # First try: transform with preprocessor
            X_proc = None
            if preprocessor is not None:
                try:
                    X_proc = preprocessor.transform(X_pred)
                except Exception as e:
                    print('Preprocessor transform failed:', e)
                    # Try to use processed_test file (if present in params)
                    try:
                        proc_df = safe_read_csv(processed_test)
                        if proc_df is not None:
                            X_proc = proc_df.reset_index(drop=True).iloc[:len(df_sample)]
                            print('Using processed_test.csv for predictions')
                    except Exception:
                        X_proc = None

            # If still no X_proc, try to align features heuristically
            if X_proc is None:
                # If preprocessor exposes expected feature names, try to match
                expected = None
                try:
                    if preprocessor is not None and hasattr(preprocessor, 'get_feature_names_out'):
                        expected = list(preprocessor.get_feature_names_out())
                except Exception:
                    expected = None

                if expected:
                    common = [c for c in X_pred.columns if c in expected]
                    if common:
                        X_proc = X_pred[common]
                        print('Using common columns with preprocessor expected features:', common[:10])
                else:
                    # As a last resort, if X_pred has same number of columns as model.coef_, use it directly
                    try:
                        if hasattr(model, 'coef_') and X_pred.shape[1] == len(model.coef_.ravel()):
                            X_proc = X_pred
                            print('Using X_pred directly because its shape matches model.coef_')
                    except Exception:
                        X_proc = None

            if X_proc is not None:
                try:
                    probs = model.predict_proba(X_proc)[:, 1]
                except Exception as e:
                    print('Model predict_proba failed on prepared X_proc:', e)
                    # try decision_function->scaled
                    try:
                        scores = model.decision_function(X_proc)
                        probs = (scores - scores.min()) / (scores.max() - scores.min())
                    except Exception as e2:
                        print('decision_function fallback failed:', e2)
                        probs = None
            else:
                print('Could not prepare feature matrix for probability prediction; skipping probability-based plots')
        except Exception as e:
            print('Could not compute probabilities for focused EDA:', e)

    # Now create the requested plots
    if probs is not None:
        plot_risk_score_pie(probs, figs)
        

        # Metrics by segment (requires raw_df and y_test)
        if raw_df is not None and y_test is not None and len(y_test) == len(probs):
            segment_cols = ['Category', 'Country', 'SalesChannel']
            plot_metrics_by_segment(raw_df.reset_index(drop=True).iloc[:len(probs)], probs, y_test, segment_cols, figs)

    

        # ROC and PR
        if y_test is not None and len(y_test) == len(probs):
            plot_roc_auc(y_test, probs, figs)
            plot_precision_recall_curve(y_test, probs, figs)

    # Time trends: monthly, hourly, weekly (use season_df and date_candidates)
    plot_time_trends(season_df, date_candidates, target_col, figs)

    print('EDA complete. Figures and summary saved under', report_dir)


if __name__ == '__main__':
    main()
