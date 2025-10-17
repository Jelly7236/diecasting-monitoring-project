# viz/control_plots.py
import numpy as np
import plotly.graph_objs as go

def build_univar_figure(x, mu, sd, violations, title):
    v_idx = {vi[0]-1 for vi in violations}
    colors = ['#ef4444' if i in v_idx else '#2563eb' for i in range(len(x))]
    ucl, lcl = mu + 3*sd, mu - 3*sd

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(1,len(x)+1), y=x, mode="lines+markers",
        line=dict(color='#2563eb', width=2),
        marker=dict(color=colors, size=6), name="데이터"
    ))
    fig.add_hline(y=mu,  line_color="#10b981", annotation_text="CL")
    fig.add_hline(y=ucl, line_dash="dash", line_color="#ef4444", annotation_text="UCL")
    fig.add_hline(y=lcl, line_dash="dash", line_color="#ef4444", annotation_text="LCL")
    fig.add_hrect(y0=mu-sd, y1=mu+sd,  fillcolor="#dbeafe", opacity=.25, layer="below")
    fig.add_hrect(y0=mu-2*sd, y1=mu+2*sd, fillcolor="#bfdbfe", opacity=.18, layer="below")
    fig.update_layout(template="plotly_white", height=420, hovermode="x unified",
                      title=title, xaxis_title="샘플", yaxis_title="값")
    return fig

def build_t2_figure(t2, ucl, title, viol_idx):
    colors = ['#ef4444' if i in viol_idx else '#2563eb' for i in range(len(t2))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(1,len(t2)+1), y=t2, mode="lines+markers",
        marker=dict(color=colors, size=6), line=dict(color="#2563eb", width=2), name="T²"
    ))
    fig.add_hline(y=ucl, line_dash="dash", line_color="#ef4444", annotation_text=f"UCL {ucl:.2f}")
    fig.update_layout(template="plotly_white", height=420, hovermode="x unified",
                      title=title, xaxis_title="샘플", yaxis_title="T²")
    return fig

def build_cap_hist(x, usl, lsl, mean_val, cp, cpk, title):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x, nbinsx=30, name="분포", marker_color="#2563eb", opacity=.75))
    fig.add_vline(x=usl, line_color="#ef4444", line_dash="dash", annotation_text="USL")
    fig.add_vline(x=lsl, line_color="#ef4444", line_dash="dash", annotation_text="LSL")
    fig.add_vline(x=mean_val, line_color="#10b981", annotation_text="평균")
    fig.update_layout(template="plotly_white", height=420,
                      title=f"{title} (Cp={cp:.2f}, Cpk={cpk:.2f})",
                      xaxis_title="값", yaxis_title="빈도")
    return fig
