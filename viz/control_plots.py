# viz/control_plots.py
import plotly.graph_objs as go

# 단변량 관리도
def build_univar_figure(x, mu, sd, vio, title="관리도"):
    ucl, lcl = mu + 3*sd, mu - 3*sd
    viol_idx = [v[0]-1 for v in vio]
    colors = ['red' if i in viol_idx else '#3b82f6' for i in range(len(x))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1,len(x)+1)), y=x, mode='lines+markers',
                             marker=dict(color=colors), line=dict(width=2)))
    fig.add_hline(y=mu, line_color="#10b981", annotation_text="CL", annotation_position="right")
    fig.add_hline(y=ucl, line_color="#ef4444", line_dash="dash", annotation_text="UCL")
    fig.add_hline(y=lcl, line_color="#ef4444", line_dash="dash", annotation_text="LCL")
    fig.update_layout(title=title, template="plotly_white", height=350)
    return fig


# Hotelling T² 관리도
def build_t2_figure(t2, ucl, title="Hotelling T²", viol_idx=None):
    colors = ['red' if i in (viol_idx or []) else '#3b82f6' for i in range(len(t2))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(t2)+1)), y=t2,
                             mode='lines+markers', marker=dict(color=colors)))
    fig.add_hline(y=ucl, line_color="#ef4444", line_dash="dash", annotation_text=f"UCL={ucl:.2f}")
    fig.update_layout(title=title, template="plotly_white", height=350)
    return fig


# Cp/Cpk 히스토그램
def build_cap_hist(x, usl, lsl, mean, cp, cpk, title="Cp/Cpk"):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x, nbinsx=30, name="분포", marker_color="#60a5fa"))
    fig.add_vline(x=usl, line_dash="dash", line_color="#ef4444", annotation_text="USL")
    fig.add_vline(x=lsl, line_dash="dash", line_color="#ef4444", annotation_text="LSL")
    fig.add_vline(x=mean, line_color="#10b981", annotation_text="Mean")
    fig.update_layout(title=f"{title} (Cp={cp:.2f}, Cpk={cpk:.2f})", template="plotly_white", height=350)
    return fig
