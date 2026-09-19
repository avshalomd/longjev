"""Builds the social-media pipeline graphic (1200x627) as an HTML page holding one SVG.

Palette and the dotted retro-window motif follow typesafe.ai; "Jev" is set as a text wordmark because
TypeSafe has no graphical Jev logo. Render with headless Chrome (see the command at the bottom)."""

from pathlib import Path

INK, PINK, MAGENTA, PAPER, GRAY, LINE = "#1E1E1E", "#F386A1", "#D45BB6", "#FEFEFE", "#DEDEDE", "#C4C4C4"
W, H = 1200, 627
WIN_W, WIN_H, WIN_Y, GAP, LEFT = 222, 292, 168, 64, 60
XS = [LEFT + i * (WIN_W + GAP) for i in range(4)]
TITLES = ["01  input", "02  compress", "03  decide", "04  answer"]
CAPTIONS = [("A long document", "and a question"), ("Jev scores every piece", "and keeps the best"),
            ("Jev reads the short", "version and decides"), ("A typed answer,", "with confidence")]


def window(x: int, title: str, jev: bool) -> str:
    badge = (f'<rect x="{x + WIN_W - 66}" y="{WIN_Y + 7}" width="54" height="22" rx="11" fill="{PINK}"/>'
             f'<text class="g" x="{x + WIN_W - 39}" y="{WIN_Y + 23}" text-anchor="middle" font-size="15" font-weight="800" fill="{INK}">Jev</text>') if jev else ""
    return (f'<rect x="{x + 6}" y="{WIN_Y + 8}" width="{WIN_W}" height="{WIN_H}" rx="12" fill="{INK}" opacity=".9"/>'
            f'<rect x="{x}" y="{WIN_Y}" width="{WIN_W}" height="{WIN_H}" rx="12" fill="{PINK}" stroke="{INK}" stroke-width="3"/>'
            f'<rect x="{x}" y="{WIN_Y}" width="{WIN_W}" height="{WIN_H}" rx="12" fill="url(#dots)"/>'
            f'<path d="M{x} {WIN_Y + 36} v-24 a12 12 0 0 1 12 -12 h{WIN_W - 24} a12 12 0 0 1 12 12 v24 z" fill="{INK}"/>'
            f'<text class="m" x="{x + 16}" y="{WIN_Y + 23}" font-size="13" fill="{PAPER}" letter-spacing="1.2">{title.upper()}</text>{badge}'
            f'<rect x="{x + 16}" y="{WIN_Y + 52}" width="{WIN_W - 32}" height="{WIN_H - 68}" rx="8" fill="{PAPER}" stroke="{INK}" stroke-width="2.5"/>')


def bubble(cx: int, cy: int, r: int = 26) -> str:
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{MAGENTA}" stroke="{INK}" stroke-width="3"/>'
            f'<path d="M{cx - 14} {cy + r - 6} l-8 16 l20 -10 z" fill="{MAGENTA}" stroke="{INK}" stroke-width="3" stroke-linejoin="round"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r - 2}" fill="{MAGENTA}"/>'
            f'<text class="g" x="{cx}" y="{cy + 12}" text-anchor="middle" font-size="34" font-weight="800" fill="{PAPER}">?</text>')


def station_input(x: int) -> str:
    sx, sy = x + 16, WIN_Y + 52
    pages = "".join(f'<rect x="{sx + 28 + o}" y="{sy + 22 - o}" width="92" height="150" rx="6" fill="{PAPER}" stroke="{INK}" stroke-width="3"/>' for o in (16, 8, 0))
    lines = "".join(f'<line x1="{sx + 42}" y1="{sy + 46 + i * 15}" x2="{sx + (106 if i % 3 else 94)}" y2="{sy + 46 + i * 15}" stroke="{INK}" stroke-width="3" stroke-linecap="round" opacity=".55"/>' for i in range(8))
    return (pages + lines + bubble(sx + 146, sy + 150)
            + f'<text class="m" x="{sx + 95}" y="{sy + 208}" text-anchor="middle" font-size="12.5" fill="{INK}">too long for one read</text>')


def station_compress(x: int) -> str:
    sx, sy = x + 16, WIN_Y + 52
    keep = {1, 6, 8, 15, 18}
    cells = []
    for i in range(20):
        cx, cy = sx + 15 + (i % 4) * 42, sy + 16 + (i // 4) * 34
        on = i in keep
        cells.append(f'<rect x="{cx}" y="{cy}" width="34" height="26" rx="5" fill="{MAGENTA if on else GRAY}" stroke="{INK if on else LINE}" stroke-width="{3 if on else 2}"/>')
        if on:
            cells.append(f'<path d="M{cx + 10} {cy + 13} l5 5 l9 -10" fill="none" stroke="{PAPER}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>')
    return "".join(cells) + f'<text class="m" x="{sx + 95}" y="{sy + 208}" text-anchor="middle" font-size="12.5" fill="{INK}">relevant pieces kept</text>'


def station_decide(x: int) -> str:
    sx, sy = x + 16, WIN_Y + 52
    bars = "".join(f'<rect x="{sx + 18}" y="{sy + 20 + i * 19}" width="{96 if i != 2 else 78}" height="13" rx="4" fill="{MAGENTA}" stroke="{INK}" stroke-width="2.5"/>' for i in range(5))
    tile = (f'<rect x="{sx + 18}" y="{sy + 132}" width="154" height="58" rx="10" fill="{INK}"/>'
            f'<text class="g" x="{sx + 95}" y="{sy + 173}" text-anchor="middle" font-size="38" font-weight="800" fill="{PINK}">Jev</text>')
    return (bars + bubble(sx + 148, sy + 52, 22)
            + f'<path d="M{sx + 95} {sy + 116} v10" stroke="{INK}" stroke-width="3.5" stroke-linecap="round"/><path d="M{sx + 88} {sy + 121} l7 7 l7 -7" fill="none" stroke="{INK}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>'
            + tile + f'<text class="m" x="{sx + 95}" y="{sy + 208}" text-anchor="middle" font-size="12.5" fill="{INK}">one quick call</text>')


def station_answer(x: int) -> str:
    sx, sy = x + 16, WIN_Y + 52
    return (f'<circle cx="{sx + 95}" cy="{sy + 62}" r="42" fill="{MAGENTA}" stroke="{INK}" stroke-width="3"/>'
            f'<path d="M{sx + 75} {sy + 63} l14 15 l27 -31" fill="none" stroke="{PAPER}" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<text class="g" x="{sx + 95}" y="{sy + 148}" text-anchor="middle" font-size="34" font-weight="800" fill="{INK}">Yes</text>'
            f'<rect x="{sx + 25}" y="{sy + 164}" width="140" height="12" rx="6" fill="{GRAY}" stroke="{INK}" stroke-width="2.5"/>'
            f'<rect x="{sx + 25}" y="{sy + 164}" width="127" height="12" rx="6" fill="{PINK}" stroke="{INK}" stroke-width="2.5"/>'
            f'<text class="m" x="{sx + 95}" y="{sy + 208}" text-anchor="middle" font-size="12.5" fill="{INK}">91% confident</text>')


def arrow(x: int) -> str:
    y, a, b = WIN_Y + WIN_H // 2, x + WIN_W + 16, x + WIN_W + GAP - 12
    return (f'<line x1="{a}" y1="{y}" x2="{b - 4}" y2="{y}" stroke="{INK}" stroke-width="5" stroke-linecap="round"/>'
            f'<path d="M{b - 14} {y - 11} L{b} {y} L{b - 14} {y + 11}" fill="none" stroke="{INK}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>')


def build() -> str:
    stations = [station_input, station_compress, station_decide, station_answer]
    body = "".join(window(x, TITLES[i], i in (1, 2)) + stations[i](x) for i, x in enumerate(XS))
    body += "".join(arrow(x) for x in XS[:3])
    body += "".join(f'<text class="g" x="{x + WIN_W / 2}" y="{WIN_Y + WIN_H + 40 + j * 25}" text-anchor="middle" font-size="19" font-weight="700" fill="{INK}">{line}</text>'
                    for x, pair in zip(XS, CAPTIONS) for j, line in enumerate(pair))
    return f'''<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@500;700;800&family=JetBrains+Mono:wght@500&display=block">
<style>html,body{{margin:0;background:{PAPER}}} svg{{display:block}} .g{{font-family:"Schibsted Grotesk",Helvetica,Arial,sans-serif}} .m{{font-family:"JetBrains Mono",Menlo,monospace;font-weight:500}}</style>
</head><body><svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#EAF2FB"/><stop offset=".55" stop-color="{PAPER}"/><stop offset="1" stop-color="#FCE4EB"/></linearGradient>
<filter id="blur" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="38"/></filter>
<pattern id="dots" width="9" height="9" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1.1" fill="{INK}" opacity=".2"/></pattern>
</defs>
<rect width="{W}" height="{H}" fill="url(#sky)"/>
<g filter="url(#blur)" opacity=".55"><circle cx="1090" cy="10" r="130" fill="{MAGENTA}"/><circle cx="960" cy="-30" r="90" fill="{PINK}"/><circle cx="40" cy="620" r="120" fill="{PINK}"/><circle cx="1180" cy="600" r="80" fill="{MAGENTA}"/></g>
<text class="g" x="{LEFT}" y="84" font-size="46" font-weight="800" fill="{INK}" letter-spacing="-1">Long documents for Jev</text>
<text class="m" x="{LEFT + 2}" y="120" font-size="17" fill="{INK}" opacity=".8">Jev reads every piece, keeps what matters, then answers.</text>
{body}
<text class="m" x="{LEFT + 2}" y="{H - 24}" font-size="13" fill="{INK}" opacity=".7">An independent experiment. Not affiliated with TypeSafe AI.</text>
</svg></body></html>'''


if __name__ == "__main__":
    out = Path(__file__).with_name("longjev_pipeline.html")
    out.write_text(build())
    print(out)
    # "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --hide-scrollbars \
    #   --window-size=1200,627 --force-device-scale-factor=2 --virtual-time-budget=6000 \
    #   --screenshot=results/social/longjev_pipeline.png file://$PWD/results/social/longjev_pipeline.html
