import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] = False


def plot_live(df, cols):
    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor='white')
    ax.set_facecolor('#fafafa')

    if df is None or df.empty or not cols:
        ax.text(0.5, 0.5, "▶ 스트리밍을 시작하세요", ha="center", va="center",
                fontsize=15, color='#9ca3af', fontweight='500')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        plt.tight_layout(); return fig

    # X축: 데이터의 실제 시간(datetime)
    x = df["datetime"] if "datetime" in df.columns else df.index
    if "datetime" in df.columns:
        try:
            x = pd.to_datetime(x)
        except Exception:
            pass

    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#14b8a6']
    for i, col in enumerate(cols):
        if col in df.columns:
            ax.plot(x, df[col], label=col, lw=2.4,
                    color=colors[i % len(colors)], alpha=0.95)

    ax.legend(loc='upper left', frameon=True, fancybox=True,
              fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color='#d1d5db')
    ax.set_xlabel('시간', fontsize=11, fontweight='500', color='#374151')
    ax.set_ylabel('센서 값', fontsize=11, fontweight='500', color='#374151')
    ax.tick_params(labelsize=9, colors='#6b7280')

    # 날짜 포맷터/로케이터
    if "datetime" in df.columns:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#d1d5db')
        ax.spines[spine].set_linewidth(1)

    plt.tight_layout()
    return fig


def plot_oee(metrics):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
    labels = ["가동률", "성능", "품질", "OEE"]
    values = [metrics.get("availability", 0), metrics.get("performance", 0),
              metrics.get("quality", 0), metrics.get("oee", 0)]
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('비율', fontsize=11)
    ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f'{h:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout(); return fig


def plot_mold_pie(data):
    fig, ax = plt.subplots(figsize=(3.5, 3.5), facecolor='white')
    sizes = [data.get("good", 0), data.get("defect", 0)]
    colors = ['#10b981', '#ef4444']; labels = ['양품', '불량']
    if sum(sizes) == 0:
        ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center",
                fontsize=11, color='#6b7280'); ax.axis("off")
    else:
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10}
        )
        for autotext in autotexts:
            autotext.set_color('white'); autotext.set_fontweight('bold')
    plt.tight_layout(); return fig


def plot_mold_ratio(molds):
    labels = list(molds.keys())
    sizes = [molds[m]["good"] + molds[m]["defect"] for m in labels]
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    if sum(sizes) == 0:
        ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center",
                fontsize=12, color='#6b7280'); ax.axis("off")
    else:
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            startangle=90, colors=colors[:len(labels)],
            textprops={'fontsize': 10}
        )
        for autotext in autotexts:
            autotext.set_color('white'); autotext.set_fontweight('bold')
    plt.tight_layout(); return fig
