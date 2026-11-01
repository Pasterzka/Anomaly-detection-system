import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_for_manual_labeling(df, stock_name):
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1,
        subplot_titles=(f'{stock_name} - Cena akcji (Close) i Trend (SMA50)', 'Stopa Zwrotu (Return)')
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df['close'], name='Cena (Close)', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA50'], name='Trend (SMA50)', line=dict(color='orange')),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=df.index, y=df['Return'], name='Return', marker_color='red'),
        row=2, col=1
    )

    fig.update_layout(
        height=800, 
        title_text="Narzędzie do ręcznego etykietowania anomalii", 
        hovermode="x unified"
    )
    
    fig.show()

def export_data_for_labeling(df, filename="intel_labeling_dataset.csv"):
    print("[INFO] Exporting data for manual labeling...")
    
    export_df = df[['close', 'SMA50', 'Return']].copy()
    
    export_df['Manual_Is_Anomaly'] = 0 
    export_df['Manual_Explanation'] = ""
    
    export_df['ZScore_Point'] = 0
    export_df['XAI_ZScore_Point'] = ""
    export_df['ZScore_Collective'] = 0
    export_df['XAI_ZScore_Collective'] = ""
    export_df['ZScore_Contextual'] = 0
    export_df['XAI_ZScore_Contextual'] = ""


    export_df['IQR_Point'] = 0
    export_df['XAI_IQR_Point'] = ""
    export_df['IQR_Collective'] = 0
    export_df['XAI_IQR_Collective'] = ""
    export_df['IQR_Contextual'] = 0
    export_df['XAI_IQR_Contextual'] = ""


    export_df['DBSCAN_Point'] = 0
    export_df['XAI_DBSCAN_Point'] = ""
    export_df['DBSCAN_Collective'] = 0
    export_df['XAI_DBSCAN_Collective'] = ""
    export_df['DBSCAN_Contextual'] = 0
    export_df['XAI_DBSCAN_Contextual'] = ""


    export_df['LOF_Point'] = 0
    export_df['XAI_LOF_Point'] = ""
    export_df['LOF_Collective'] = 0
    export_df['XAI_LOF_Collective'] = ""
    export_df['LOF_Contextual'] = 0
    export_df['XAI_LOF_Contextual'] = ""
    

    export_df.to_csv(filename)
    print(f"[SUCCESS] Saved dataset to {filename}")