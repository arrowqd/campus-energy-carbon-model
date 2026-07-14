from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"


@dataclass(frozen=True)
class Device:
    name: str
    quantity: float
    power_w: float
    daily_hours: float
    annual_days: float
    load_factor: float

    @property
    def annual_kwh(self) -> float:
        return (
            self.quantity
            * self.power_w
            * self.daily_hours
            * self.annual_days
            * self.load_factor
            / 1000
        )


def read_devices() -> list[Device]:
    with (DATA_DIR / "device_inputs.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        return [
            Device(
                name=row["device_type"],
                quantity=float(row["quantity"]),
                power_w=float(row["power_w"]),
                daily_hours=float(row["daily_hours"]),
                annual_days=float(row["annual_days"]),
                load_factor=float(row["load_factor"]),
            )
            for row in rows
        ]


def calculate() -> tuple[list[dict[str, float | str]], dict[str, float], dict[str, float]]:
    devices = read_devices()
    config = json.loads((DATA_DIR / "scenario_inputs.json").read_text(encoding="utf-8"))
    price = config["electricity_price_cny_per_kwh"]
    factor = config["carbon_factor_kgco2e_per_kwh"]
    years = config["evaluation_years"]
    schemes = config["schemes"]
    energy = {device.name: device.annual_kwh for device in devices}

    baseline_kwh = sum(energy.values())
    baseline_cost = baseline_kwh * price
    baseline_tco2e = baseline_kwh * factor / 1000

    led_kwh = energy[schemes["led"]["target"]] * schemes["led"]["saving_rate"]
    ac_kwh = energy[schemes["ac"]["target"]] * schemes["ac"]["saving_rate"]
    shiftable_kwh = sum(energy[name] for name in schemes["tou"]["targets"])
    tou_shift_kwh = shiftable_kwh * schemes["tou"]["shift_rate"]
    tou_saving = tou_shift_kwh * (
        schemes["tou"]["peak_price_cny_per_kwh"]
        - schemes["tou"]["valley_price_cny_per_kwh"]
    )
    pv_kwh = baseline_kwh * schemes["pv"]["replacement_rate"]

    scheme_rows = []
    for key, saved_kwh, shifted_kwh, annual_saving in [
        ("led", led_kwh, 0.0, led_kwh * price),
        ("ac", ac_kwh, 0.0, ac_kwh * price),
        ("tou", 0.0, tou_shift_kwh, tou_saving),
        ("pv", pv_kwh, 0.0, pv_kwh * price),
    ]:
        investment = schemes[key]["investment_cny"]
        scheme_rows.append(
            {
                "scheme": schemes[key]["name"],
                "annual_energy_saving_kwh": saved_kwh,
                "annual_shifted_energy_kwh": shifted_kwh,
                "annual_cost_saving_cny": annual_saving,
                "annual_emission_reduction_tco2e": saved_kwh * factor / 1000,
                "investment_cny": investment,
                "simple_payback_years": investment / annual_saving,
                "ten_year_net_benefit_cny": annual_saving * years - investment,
            }
        )

    post_efficiency_kwh = baseline_kwh - led_kwh - ac_kwh
    portfolio_pv_kwh = post_efficiency_kwh * schemes["pv"]["replacement_rate"]
    grid_kwh = post_efficiency_kwh - portfolio_pv_kwh
    portfolio_shift_kwh = (
        shiftable_kwh
        * (1 - schemes["pv"]["replacement_rate"])
        * schemes["tou"]["shift_rate"]
    )
    portfolio_tou_saving = portfolio_shift_kwh * (
        schemes["tou"]["peak_price_cny_per_kwh"]
        - schemes["tou"]["valley_price_cny_per_kwh"]
    )
    portfolio_cost = grid_kwh * price - portfolio_tou_saving
    portfolio_saving = baseline_cost - portfolio_cost
    portfolio_investment = sum(s["investment_cny"] for s in schemes.values())
    portfolio = {
        "annual_energy_saving_kwh": baseline_kwh - grid_kwh,
        "annual_shifted_energy_kwh": portfolio_shift_kwh,
        "annual_cost_saving_cny": portfolio_saving,
        "annual_emission_reduction_tco2e": (baseline_kwh - grid_kwh) * factor / 1000,
        "investment_cny": portfolio_investment,
        "simple_payback_years": portfolio_investment / portfolio_saving,
        "ten_year_net_benefit_cny": portfolio_saving * years - portfolio_investment,
        "grid_reduction_rate": (baseline_kwh - grid_kwh) / baseline_kwh,
    }

    baseline = {
        "annual_energy_kwh": baseline_kwh,
        "annual_cost_cny": baseline_cost,
        "annual_emissions_tco2e": baseline_tco2e,
    }

    assert abs(baseline_kwh - 182160.0) < 1e-6
    assert all(row["simple_payback_years"] > 0 for row in scheme_rows)
    assert 0 < portfolio["grid_reduction_rate"] < 1
    return scheme_rows, baseline, portfolio


def write_results(
    scheme_rows: list[dict[str, float | str]],
    baseline: dict[str, float],
    portfolio: dict[str, float],
) -> None:
    fieldnames = list(scheme_rows[0].keys())
    with (DATA_DIR / "scheme_results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scheme_rows)

    with (DATA_DIR / "summary_results.json").open("w", encoding="utf-8") as file:
        json.dump({"baseline": baseline, "portfolio": portfolio}, file, ensure_ascii=False, indent=2)


def build_figures(scheme_rows: list[dict[str, float | str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        build_figures_with_pillow(scheme_rows)
        print("提示：未检测到 Matplotlib，已使用 Pillow 生成简化图；安装 requirements.txt 后可生成标准图表。")
        return

    FIGURES_DIR.mkdir(exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    devices = read_devices()
    labels = [device.name for device in devices]
    values = [device.annual_kwh for device in devices]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    bars = ax.bar(labels, values, color=["#2563EB", "#0EA5E9", "#22C55E", "#94A3B8"])
    ax.set_title("图书馆基准年用电结构")
    ax.set_ylabel("年用电量（kWh）")
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "energy_structure.png", bbox_inches="tight")
    plt.close(fig)

    names = [str(row["scheme"]) for row in scheme_rows]
    savings = [float(row["annual_cost_saving_cny"]) for row in scheme_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    bars = ax.bar(names, savings, color="#2563EB")
    ax.set_title("单项改造方案年节省电费")
    ax.set_ylabel("年节省电费（元）")
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "scheme_cost_saving.png", bbox_inches="tight")
    plt.close(fig)

    paybacks = [float(row["simple_payback_years"]) for row in scheme_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    bars = ax.bar(names, paybacks, color=["#0EA5E9", "#22C55E", "#F59E0B", "#64748B"])
    ax.set_title("单项改造方案静态投资回收期")
    ax.set_ylabel("回收期（年）")
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "scheme_payback.png", bbox_inches="tight")
    plt.close(fig)


def build_figures_with_pillow(scheme_rows: list[dict[str, float | str]]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    FIGURES_DIR.mkdir(exist_ok=True)
    font_candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    font_path = next((item for item in font_candidates if item.exists()), None)

    def font(size: int, bold: bool = False):
        if font_path:
            bold_path = Path("C:/Windows/Fonts/msyhbd.ttc") if bold else font_path
            return ImageFont.truetype(str(bold_path if bold_path.exists() else font_path), size)
        return ImageFont.load_default()

    def bar_chart(title: str, labels: list[str], values: list[float], ylabel: str, colors: list[str], filename: str, decimals: int = 0) -> None:
        width, height = 1280, 720
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((60, 35), title, fill="#17324D", font=font(42, True))
        left, top, right, bottom = 110, 125, 1210, 590
        draw.line((left, top, left, bottom), fill="#94A3B8", width=2)
        draw.line((left, bottom, right, bottom), fill="#94A3B8", width=2)
        maximum = max(values) * 1.15 if values else 1
        slot = (right - left) / max(len(values), 1)
        bar_width = slot * 0.55
        for index, (label, value) in enumerate(zip(labels, values)):
            x1 = left + index * slot + slot * 0.225
            x2 = x1 + bar_width
            y1 = bottom - (value / maximum) * (bottom - top)
            draw.rounded_rectangle((x1, y1, x2, bottom), radius=8, fill=colors[index % len(colors)])
            value_text = f"{value:.{decimals}f}"
            box = draw.textbbox((0, 0), value_text, font=font(23, True))
            draw.text(((x1 + x2 - (box[2] - box[0])) / 2, y1 - 34), value_text, fill="#111827", font=font(23, True))
            label_box = draw.textbbox((0, 0), label, font=font(22))
            draw.text(((x1 + x2 - (label_box[2] - label_box[0])) / 2, bottom + 18), label, fill="#334155", font=font(22))
        draw.text((30, top + 120), ylabel, fill="#64748B", font=font(20))
        image.save(FIGURES_DIR / filename)

    devices = read_devices()
    bar_chart(
        "图书馆基准年用电结构",
        [device.name for device in devices],
        [device.annual_kwh for device in devices],
        "年用电量（kWh）",
        ["#2563EB", "#0EA5E9", "#22C55E", "#94A3B8"],
        "energy_structure.png",
    )
    names = [str(row["scheme"]) for row in scheme_rows]
    bar_chart(
        "单项改造方案年节省电费",
        names,
        [float(row["annual_cost_saving_cny"]) for row in scheme_rows],
        "年节省电费（元）",
        ["#2563EB"],
        "scheme_cost_saving.png",
    )
    bar_chart(
        "单项改造方案静态投资回收期",
        names,
        [float(row["simple_payback_years"]) for row in scheme_rows],
        "回收期（年）",
        ["#0EA5E9", "#22C55E", "#F59E0B", "#64748B"],
        "scheme_payback.png",
        decimals=2,
    )


def main() -> None:
    scheme_rows, baseline, portfolio = calculate()
    write_results(scheme_rows, baseline, portfolio)
    build_figures(scheme_rows)
    print(f"基准年用电量: {baseline['annual_energy_kwh']:,.0f} kWh")
    print(f"基准年碳排放: {baseline['annual_emissions_tco2e']:.2f} tCO2e")
    print(f"组合方案购电量降幅: {portfolio['grid_reduction_rate']:.1%}")
    print(f"组合方案回收期: {portfolio['simple_payback_years']:.2f} 年")


if __name__ == "__main__":
    main()
