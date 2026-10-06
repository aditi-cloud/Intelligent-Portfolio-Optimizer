"""Small self-contained SVG charts of supplied results; no financial calculations."""

from html import escape
from math import isfinite


def line_chart(title: str, series: dict[str, list[tuple[float, float]]], *, x_label: str, y_label: str) -> str:
    points = [point for values in series.values() for point in values]
    if not points or not all(isfinite(x) and isfinite(y) for x, y in points):
        raise ValueError("Chart points must be finite and nonempty")
    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    ymin, ymax = min(y for _, y in points), max(y for _, y in points)
    if xmin == xmax:
        xmin, xmax = xmin - .5, xmax + .5
    if ymin == ymax:
        padding = max(abs(ymin) * .05, 1)
        ymin, ymax = ymin - padding, ymax + padding
    def position(x, y):
        return 65 + 600 * (x - xmin) / (xmax - xmin), 270 - 220 * (y - ymin) / (ymax - ymin)
    colors = ["#2563eb", "#059669", "#d97706", "#9333ea"]
    parts = [f'<figure><figcaption>{escape(title)}</figcaption><svg viewBox="0 0 730 350" role="img" aria-label="{escape(title, quote=True)}">',
             '<path d="M65 50V270H665" fill="none" stroke="#64748b"/>']
    for step in range(5):
        x = xmin + (xmax - xmin) * step / 4
        y = ymin + (ymax - ymin) * step / 4
        px, py = position(x, y)
        parts.append(f'<text x="{px:.2f}" y="290" text-anchor="middle">{x:.3g}</text>')
        parts.append(f'<text x="60" y="{py:.2f}" text-anchor="end">{y:.3g}</text>')
    for index, (name, values) in enumerate(series.items()):
        color = colors[index % len(colors)]
        coordinates = " ".join(f"{px:.2f},{py:.2f}" for px, py in (position(x, y) for x, y in values))
        parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
        for px, py in (position(x, y) for x, y in values):
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2" fill="{color}"/>')
        parts.append(f'<text x="{65 + index * 160}" y="340" fill="{color}">{escape(name)}</text>')
    parts.extend([f'<text x="365" y="315" text-anchor="middle">{escape(x_label)}</text>',
                  f'<text x="65" y="25">{escape(y_label)}</text>', '</svg></figure>'])
    return "".join(parts)
