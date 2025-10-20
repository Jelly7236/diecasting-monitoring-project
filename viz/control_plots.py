import plotly.graph_objs as go

# ------------------------
# 다변량 T² 관리도
# ------------------------
def plot_t2_chart(t2_values, ucl, n_samples):
    colors = ['red' if val > ucl else '#3b82f6' for val in t2_values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(t2_values) + 1)),
        y=t2_values,
        mode='lines+markers',
        name='T² 통계량',
        line=dict(color='#3b82f6', width=2),
        marker=dict(color=colors, size=5)
    ))

    fig.add_hline(
        y=ucl, line_dash="dash", line_color="#ef4444",
        annotation_text=f"UCL ({ucl:.2f})", annotation_position="right"
    )

    fig.update_layout(
        title=f"Hotelling T² (n={n_samples})",
        xaxis_title="샘플 번호",
        yaxis_title="T² 값",
        template="plotly_white",
        height=350,
        hovermode='x unified'
    )
    return fig


# ------------------------
# 단변량 관리도 (모달용)
# ------------------------
def plot_univariate_chart(x, mu, ucl, lcl, sd, vio_idx, var):
    colors = ['red' if i+1 in vio_idx else '#3b82f6' for i in range(len(x))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(x) + 1)),
        y=x,
        mode='lines+markers',
        name='데이터',
        line=dict(color='#3b82f6', width=2),
        marker=dict(color=colors, size=5)
    ))

    fig.add_hline(y=mu, line_dash="solid", line_color="#10b981",
                  annotation_text="CL", annotation_position="right")
    fig.add_hline(y=ucl, line_dash="dash", line_color="#ef4444",
                  annotation_text="UCL", annotation_position="right")
    fig.add_hline(y=lcl, line_dash="dash", line_color="#ef4444",
                  annotation_text="LCL", annotation_position="right")

    fig.add_hrect(y0=mu-sd, y1=mu+sd, fillcolor="#dbeafe", opacity=0.2)
    fig.add_hrect(y0=mu-2*sd, y1=mu+2*sd, fillcolor="#bfdbfe", opacity=0.15)

    fig.update_layout(
        title=f"{var} 관리도 (n={len(x)})",
        xaxis_title="샘플 번호",
        yaxis_title="측정값",
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )
    return fig
