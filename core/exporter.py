"""
Modul Export Laporan Hasil Survei SDI PKRMS.
Mendukung ekspor ke CSV, Excel (.xlsx) dengan pemformatan rapi, dan Laporan HTML/Cetak.
"""
import io
import pandas as pd
from typing import List, Dict, Any
from .segmentation import RoadSegment


def segments_to_dataframe(segments: List[RoadSegment], include_details: bool = True) -> pd.DataFrame:
    """Mengubah daftar RoadSegment menjadi Pandas DataFrame terstruktur."""
    data = []
    for s in segments:
        row = {
            "Nama Jalan": s.road_name,
            "STA Awal (m)": s.sta_start,
            "STA Akhir (m)": s.sta_end,
            "Panjang (m)": s.length_m,
            "Nilai SDI": s.sdi_value,
            "Jenis Perkerasan": s.pavement_type,
            "Kondisi": s.condition,
            "Saran Perbaikan": s.recommendation,
        }
        if include_details:
            row.update({
                "% Luas Retak": s.crack_area_pct,
                "Lebar Retak (mm)": s.avg_crack_width_mm,
                "Lubang/km": s.potholes_per_km,
                "Jumlah Lubang Unik": s.unique_potholes,
                "Kedalaman Alur (cm)": s.rut_depth_cm,
                "Total Luas Retak (m2)": s.total_crack_area_m2,
                "Status/Flag": s.flag,
            })
        data.append(row)

    return pd.DataFrame(data)


def export_to_csv(segments: List[RoadSegment]) -> str:
    """Mengekspor hasil evaluasi ke string CSV format UTF-8."""
    df = segments_to_dataframe(segments, include_details=True)
    return df.to_csv(index=False)


def export_to_excel(segments: List[RoadSegment], road_info: Dict[str, Any]) -> bytes:
    """
    Mengekspor hasil survei ke file Excel (.xlsx) dengan styling profesional:
    - Header berlatar warna biru dongker
    - Pewarnaan sel kolom Kondisi sesuai standar Bina Marga (Hijau, Kuning, Oranye, Merah)
    - Auto-adjust lebar kolom
    - Sheet Ringkasan KPI dan Sheet Data Segmen
    """
    output = io.BytesIO()
    df_segments = segments_to_dataframe(segments, include_details=True)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: Data Segmen
        df_segments.to_excel(writer, sheet_name="Data Segmen SDI", index=False)

        # Sheet 2: Ringkasan Survei
        total_len = sum(s.length_m for s in segments)
        weighted_sdi = (
            sum(s.sdi_value * s.length_m for s in segments) / total_len
            if total_len > 0 else 0.0
        )
        mantap_len = sum(s.length_m for s in segments if s.condition in ["Baik", "Sedang"])
        kemantapan_pct = (mantap_len / total_len * 100.0) if total_len > 0 else 0.0

        summary_data = [
            {"Parameter": "Nama Ruas Jalan", "Nilai": road_info.get("road_name", "-")},
            {"Parameter": "STA Awal", "Nilai": f"{road_info.get('sta_start', 0)} m"},
            {"Parameter": "Panjang Total Ruas", "Nilai": f"{total_len:.1f} m ({total_len/1000:.2f} km)"},
            {"Parameter": "Lebar Jalan Rata-rata", "Nilai": f"{road_info.get('road_width_m', 6.0)} m"},
            {"Parameter": "SDI Rata-rata Terbobot Panjang", "Nilai": f"{weighted_sdi:.1f}"},
            {"Parameter": "Tingkat Kemantapan Jalan (Baik + Sedang)", "Nilai": f"{kemantapan_pct:.1f}%"},
            {"Parameter": "Jumlah Segmen", "Nilai": len(segments)},
            {"Parameter": "Standar Acuan", "Nilai": "SE Dirjen Bina Marga No. 22/SE/Db/2021 (PKRMS)"},
        ]
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name="Ringkasan Survei", index=False)

        # Styling workbook
        wb = writer.book
        ws_seg = wb["Data Segmen SDI"]

        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        center_align = Alignment(horizontal="center", vertical="center")
        thin_border = Border(
            left=Side(style="thin", color="D3D3D3"),
            right=Side(style="thin", color="D3D3D3"),
            top=Side(style="thin", color="D3D3D3"),
            bottom=Side(style="thin", color="D3D3D3")
        )

        condition_fills = {
            "Baik": PatternFill(start_color="D4EDDA", end_color="D4EDDA", fill_type="solid"),
            "Sedang": PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid"),
            "Rusak Ringan": PatternFill(start_color="FFE5D0", end_color="FFE5D0", fill_type="solid"),
            "Rusak Berat": PatternFill(start_color="F8D7DA", end_color="F8D7DA", fill_type="solid"),
        }

        # Format Header Data Segmen
        for col_idx in range(1, len(df_segments.columns) + 1):
            cell = ws_seg.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center_align

        # Format Rows
        kondisi_col_idx = df_segments.columns.get_loc("Kondisi") + 1
        for row_idx in range(2, len(df_segments) + 2):
            cond_val = ws_seg.cell(row=row_idx, column=kondisi_col_idx).value
            fill_to_apply = condition_fills.get(cond_val, None)

            for col_idx in range(1, len(df_segments.columns) + 1):
                cell = ws_seg.cell(row=row_idx, column=col_idx)
                cell.border = thin_border
                if fill_to_apply and col_idx == kondisi_col_idx:
                    cell.fill = fill_to_apply
                    cell.font = Font(bold=True)

        # Auto width
        for col in ws_seg.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws_seg.column_dimensions[col_letter].width = max(max_len + 3, 12)

    return output.getvalue()


def export_to_html_report(segments: List[RoadSegment], road_info: Dict[str, Any]) -> str:
    """Menghasilkan laporan HTML ringkas yang rapi dan siap dicetak ke PDF via browser."""
    total_len = sum(s.length_m for s in segments)
    weighted_sdi = (
        sum(s.sdi_value * s.length_m for s in segments) / total_len
        if total_len > 0 else 0.0
    )
    mantap_len = sum(s.length_m for s in segments if s.condition in ["Baik", "Sedang"])
    kemantapan_pct = (mantap_len / total_len * 100.0) if total_len > 0 else 0.0

    rows_html = ""
    for s in segments:
        badge_style = {
            "Baik": "background-color: #2ECC71; color: white;",
            "Sedang": "background-color: #F1C40F; color: black;",
            "Rusak Ringan": "background-color: #E67E22; color: white;",
            "Rusak Berat": "background-color: #E74C3C; color: white;",
        }.get(s.condition, "")

        rows_html += f"""
        <tr>
            <td>{s.road_name}</td>
            <td style="text-align: center;">{s.sta_start:.0f}</td>
            <td style="text-align: center;">{s.sta_end:.0f}</td>
            <td style="text-align: center;"><strong>{s.sdi_value:.1f}</strong></td>
            <td style="text-align: center;">{s.pavement_type}</td>
            <td style="text-align: center;"><span style="padding: 4px 8px; border-radius: 4px; font-weight: bold; {badge_style}">{s.condition}</span></td>
            <td>{s.recommendation}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <title>Laporan Survei Kondisi Jalan - Metode SDI PKRMS</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; color: #333; }}
            h1, h2, h3 {{ color: #1F497D; }}
            .header-box {{ border-bottom: 2px solid #1F497D; padding-bottom: 15px; margin-bottom: 20px; }}
            .kpi-container {{ display: flex; gap: 20px; margin-bottom: 25px; }}
            .kpi-card {{ flex: 1; padding: 15px; background: #f8f9fa; border-left: 4px solid #1F497D; border-radius: 4px; }}
            .kpi-title {{ font-size: 13px; color: #666; }}
            .kpi-val {{ font-size: 22px; font-weight: bold; color: #1F497D; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; font-size: 13px; }}
            th {{ background-color: #1F497D; color: white; text-align: center; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            @media print {{
                body {{ margin: 10px; }}
                button {{ display: none; }}
            }}
        </style>
    </head>
    <body>
        <div class="header-box">
            <h2>LAPORAN EVALUASI SURVEI KONDISI JALAN (SDI)</h2>
            <p>Sesuai Manual PKRMS (SE Dirjen Bina Marga No. 22/SE/Db/2021, Manual No. 04/M/BM/2021 Bab 4.2.5)</p>
        </div>

        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-title">Ruas Jalan</div>
                <div class="kpi-val">{road_info.get('road_name', 'Survei')}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Panjang Total</div>
                <div class="kpi-val">{total_len:.1f} m</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">SDI Rata-rata Terbobot</div>
                <div class="kpi-val">{weighted_sdi:.1f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Tingkat Kemantapan</div>
                <div class="kpi-val">{kemantapan_pct:.1f}%</div>
            </div>
        </div>

        <h3>Tabel Rekapitulasi Kondisi Jalan per Segmen</h3>
        <table>
            <thead>
                <tr>
                    <th>Nama Jalan</th>
                    <th>STA Awal</th>
                    <th>STA Akhir</th>
                    <th>Nilai SDI</th>
                    <th>Jenis Perkerasan</th>
                    <th>Kondisi</th>
                    <th>Saran Perbaikan</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>

        <div style="margin-top: 30px; font-size: 11px; color: #777;">
            Dicetak otomatis oleh Sistem Dashboard SDI PKRMS • Direktorat Jenderal Bina Marga
        </div>
    </body>
    </html>
    """
    return html

