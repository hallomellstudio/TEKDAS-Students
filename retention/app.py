"""Streamlit dashboard for a student-retention early-warning prototype.

Run from this directory after preparing the model:
    streamlit run app.py

The dashboard is intentionally support-oriented.  A risk score helps staff
prioritise outreach; it is not an automated decision about a student.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from llm import (
    build_prompt,
    generate_retention_recommendation,
    generate_rule_based_recommendation,
    risk_level,
)


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
PREDICTIONS_PATH = ARTIFACT_DIR / "predictions.csv"
FEATURE_IMPORTANCE_PATH = ARTIFACT_DIR / "feature_importance.csv"

PROBABILITY_CANDIDATES = ("dropout_probability", "risk_probability", "probability")
ID_CANDIDATES = ("student_id", "row_id", "record_id")
OUTCOME_CANDIDATES = ("outcome", "Target", "target", "actual_label")

COLUMN_LABELS = {
    "row_id": "ID pseudonim",
    "Marital status": "Status perkawinan (kode)",
    "Application mode": "Jalur pendaftaran (kode)",
    "Application order": "Urutan pilihan pendaftaran",
    "Course": "Program studi (kode)",
    "Daytime/evening attendance": "Waktu kuliah (kode)",
    "Previous qualification": "Kualifikasi sebelumnya (kode)",
    "Nacionality": "Kewarganegaraan (kode)",
    "Mother's qualification": "Pendidikan ibu (kode)",
    "Father's qualification": "Pendidikan ayah (kode)",
    "Mother's occupation": "Pekerjaan ibu (kode)",
    "Father's occupation": "Pekerjaan ayah (kode)",
    "Displaced": "Status displaced (kode)",
    "Educational special needs": "Kebutuhan pendidikan khusus (kode)",
    "Debtor": "Status debitur (kode)",
    "Tuition fees up to date": "Biaya kuliah terkini (kode)",
    "Gender": "Gender (kode)",
    "Scholarship holder": "Penerima beasiswa (kode)",
    "Age at enrollment": "Usia saat masuk",
    "International": "Mahasiswa internasional (kode)",
    "Curricular units 1st sem (credited)": "SKS semester 1 yang dikreditkan",
    "Curricular units 1st sem (enrolled)": "Mata kuliah semester 1 diambil",
    "Curricular units 1st sem (evaluations)": "Mata kuliah semester 1 dievaluasi",
    "Curricular units 1st sem (approved)": "Mata kuliah semester 1 lulus",
    "Curricular units 1st sem (grade)": "Rata-rata nilai semester 1",
    "Curricular units 1st sem (without evaluations)": "Mata kuliah semester 1 tanpa evaluasi",
    "Unemployment rate": "Tingkat pengangguran",
    "Inflation rate": "Inflasi",
    "GDP": "GDP",
}


def _first_present(columns: list[str] | pd.Index, candidates: tuple[str, ...]) -> str | None:
    available = set(columns)
    return next((name for name in candidates if name in available), None)


def _friendly_name(column: str) -> str:
    return COLUMN_LABELS.get(column, column)


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _metric_value(metrics: dict[str, Any], name: str) -> float | None:
    """Find a metric whether it is stored at top level or below ``model``."""
    for container in (metrics, metrics.get("model", {}), metrics.get("evaluation", {})):
        value = container.get(name) if isinstance(container, dict) else None
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


@st.cache_data
def load_outputs() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    predictions = pd.read_csv(PREDICTIONS_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    importance = (
        pd.read_csv(FEATURE_IMPORTANCE_PATH)
        if FEATURE_IMPORTANCE_PATH.exists()
        else pd.DataFrame()
    )
    return predictions, metrics, importance


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def probability_column(frame: pd.DataFrame) -> str | None:
    return _first_present(frame.columns, PROBABILITY_CANDIDATES)


def make_profile(record: pd.Series, feature_columns: list[str]) -> dict[str, Any]:
    return {
        column: record[column]
        for column in feature_columns
        if column in record.index
    }


def draw_profile(profile: dict[str, Any]) -> None:
    rows = [
        {"Indikator": _friendly_name(column), "Nilai": _format_value(value)}
        for column, value in profile.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def score_frame_if_needed(
    frame: pd.DataFrame, model: Any, feature_columns: list[str]
) -> tuple[pd.DataFrame, str]:
    """Use saved probabilities when available, otherwise score the supplied rows."""
    result = frame.copy()
    existing = probability_column(result)
    if existing:
        result[existing] = pd.to_numeric(result[existing], errors="coerce")
        return result, existing

    missing = sorted(set(feature_columns) - set(result.columns))
    if missing:
        raise ValueError(
            "Output prediksi tidak memuat probabilitas maupun fitur yang diperlukan: "
            + ", ".join(missing)
        )
    classes = list(model.classes_)
    dropout_index = classes.index(1) if 1 in classes else len(classes) - 1
    result["dropout_probability"] = model.predict_proba(result[feature_columns])[:, dropout_index]
    return result, "dropout_probability"


def main() -> None:
    st.set_page_config(
        page_title="Student Retention Early Warning",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("🎓 Student Retention Early Warning")
    st.caption(
        "Prototipe untuk memprioritaskan dukungan mahasiswa setelah semester 1—"
        "bukan sistem keputusan otomatis."
    )

    missing_files = [
        path.name
        for path in (MODEL_PATH, METRICS_PATH, PREDICTIONS_PATH)
        if not path.exists()
    ]
    if missing_files:
        st.error("Artifact model belum tersedia: " + ", ".join(missing_files))
        st.code(
            "cd retention\npython data_prep.py\npython train.py\nstreamlit run app.py",
            language="bash",
        )
        st.stop()

    try:
        predictions, metrics, feature_importance = load_outputs()
        model = load_model()
    except Exception as exc:  # Helpful startup message rather than a Streamlit traceback.
        st.exception(exc)
        st.stop()

    feature_columns = metrics.get("feature_columns", [])
    if not isinstance(feature_columns, list) or not feature_columns:
        excluded = {
            "row_id", "student_id", "Target", "target", "outcome", "actual_label",
            "dropout_probability", "risk_probability", "probability", "prediction",
            "predicted_dropout", "risk_band",
        }
        feature_columns = [column for column in predictions.columns if column not in excluded]

    try:
        students, probability_name = score_frame_if_needed(predictions, model, feature_columns)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    students = students.dropna(subset=[probability_name]).copy()
    id_column = _first_present(students.columns, ID_CANDIDATES)
    if id_column is None:
        students.insert(0, "row_id", [f"STU-{index + 1:04d}" for index in range(len(students))])
        id_column = "row_id"

    st.sidebar.header("Prioritas dukungan")
    threshold = st.sidebar.slider(
        "Ambang skor risiko",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Pilih sesuai kapasitas tim untuk menindaklanjuti mahasiswa."
    )
    students["prioritas_dukungan"] = students[probability_name] >= threshold
    students["kategori_risiko"] = students[probability_name].map(risk_level)

    tab_overview, tab_cohort, tab_student, tab_action, tab_model = st.tabs([
        "Ringkasan",
        "Kohort prioritas",
        "Profil mahasiswa",
        "Rencana dukungan",
        "Model & batasan",
    ])

    with tab_overview:
        total = len(students)
        prioritized = int(students["prioritas_dukungan"].sum())
        observed_col = _first_present(students.columns, OUTCOME_CANDIDATES)
        observed_dropout = None
        if observed_col:
            observed_dropout = students[observed_col].astype(str).str.lower().eq("dropout").mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mahasiswa pada kohort evaluasi", f"{total:,}")
        c2.metric("Prioritas dukungan", f"{prioritized:,}", f"≥ {threshold:.0%}")
        c3.metric("Rata-rata risiko dropout", f"{students[probability_name].mean():.1%}")
        c4.metric(
            "Dropout historis" if observed_dropout is not None else "Ambang prioritas",
            f"{observed_dropout:.1%}" if observed_dropout is not None else f"{threshold:.0%}",
        )

        st.subheader("Distribusi risiko")
        order = ["Rendah", "Sedang", "Tinggi"]
        risk_counts = (
            students["kategori_risiko"].value_counts().reindex(order, fill_value=0)
            .rename_axis("kategori")
            .to_frame("mahasiswa")
        )
        st.bar_chart(risk_counts)

        course_column = "Course" if "Course" in students.columns else None
        if course_column:
            st.subheader("Rata-rata risiko menurut program studi (kode)")
            by_course = (
                students.groupby(course_column)[probability_name]
                .mean()
                .sort_values(ascending=False)
                .head(20)
                .rename("risiko_dropout")
            )
            st.bar_chart(by_course)
        st.info(
            "Skor risiko menunjukkan pola statistik, bukan penyebab atau kepastian. "
            "Gunakan untuk menawarkan dukungan, bukan membatasi layanan atau menghukum mahasiswa."
        )

    with tab_cohort:
        st.subheader("Daftar prioritas tindak lanjut")
        st.caption("Data ini adalah kohort historis/evaluasi untuk pembelajaran, bukan daftar operasional kampus.")
        show_only_priority = st.toggle("Tampilkan hanya prioritas dukungan", value=True)
        cohort = students.loc[students["prioritas_dukungan"]].copy() if show_only_priority else students.copy()
        visible_columns = [
            id_column, "kategori_risiko", probability_name,
            *[
                column for column in (
                    "Course", "Age at enrollment", "Debtor", "Tuition fees up to date",
                    "Scholarship holder", "Curricular units 1st sem (approved)",
                    "Curricular units 1st sem (grade)",
                ) if column in cohort.columns
            ],
        ]
        table = cohort[visible_columns].sort_values(probability_name, ascending=False).copy()
        table = table.rename(columns={column: _friendly_name(column) for column in table.columns})
        probability_label = _friendly_name(probability_name)
        if probability_label in table.columns:
            table[probability_label] = table[probability_label].map(lambda value: f"{value:.1%}")
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button(
            "Unduh daftar prioritas (CSV)",
            data=cohort[visible_columns].to_csv(index=False).encode("utf-8"),
            file_name="prioritas_student_retention.csv",
            mime="text/csv",
        )

    student_ids = students[id_column].astype(str).tolist()
    chosen_id = st.sidebar.selectbox("Pilih ID mahasiswa", student_ids)
    selected = students.loc[students[id_column].astype(str).eq(chosen_id)].iloc[0]
    selected_probability = float(selected[probability_name])
    selected_profile = make_profile(selected, feature_columns)

    with tab_student:
        st.subheader(f"Profil {chosen_id}")
        left, right = st.columns((1, 2))
        with left:
            st.metric("Risiko dropout", f"{selected_probability:.1%}")
            st.metric("Kategori risiko", risk_level(selected_probability))
            st.metric("Prioritas saat ini", "Ya" if selected_probability >= threshold else "Belum")
        with right:
            st.caption("Nilai kode kategori ditampilkan apa adanya karena dataset tidak menyertakan codebook.")
            draw_profile(selected_profile)

        observed_col = _first_present(students.columns, OUTCOME_CANDIDATES)
        if observed_col:
            with st.expander("Label historis — hanya untuk evaluasi model"):
                st.write(_format_value(selected[observed_col]))

    with tab_action:
        st.subheader("Rencana dukungan yang proporsional")
        st.caption(
            "Rekomendasi di bawah hanya memakai data yang tersedia pada profil dan skor model. "
            "Petugas yang berwenang tetap memverifikasi konteks serta persetujuan mahasiswa."
        )
        st.markdown(generate_rule_based_recommendation(selected_profile, selected_probability))

        st.divider()
        st.markdown("**Opsional: ringkasan berbantuan LLM lokal**")
        st.caption("Memerlukan Ollama aktif. Jika tidak tersedia, rekomendasi aturan di atas tetap dapat digunakan.")
        model_factors = (
            feature_importance.head(10).to_dict("records")
            if not feature_importance.empty
            else []
        )
        prompt = build_prompt(
            student_profile=selected_profile,
            dropout_probability=selected_probability,
            evidence=None,
            model_factors=model_factors,
        )
        with st.expander("Periksa prompt sebelum dikirim"):
            st.code(prompt, language="text")
        if st.button("Buat ringkasan dukungan dengan LLM", type="secondary"):
            with st.spinner("Menyusun rekomendasi berbasis evidence..."):
                answer = generate_retention_recommendation(
                    student_profile=selected_profile,
                    dropout_probability=selected_probability,
                    evidence=None,
                    model_factors=model_factors,
                )
            st.markdown(answer)

    with tab_model:
        st.subheader("Kinerja pada data uji")
        metric_columns = st.columns(4)
        metric_specs = [
            ("ROC-AUC", "roc_auc"),
            ("PR-AUC", "pr_auc"),
            ("Recall dropout", "recall"),
            ("Precision dropout", "precision"),
        ]
        for target_column, (label, metric_name) in zip(metric_columns, metric_specs):
            value = _metric_value(metrics, metric_name)
            target_column.metric(label, f"{value:.3f}" if value is not None else "—")

        if not feature_importance.empty:
            st.subheader("Faktor model secara global")
            importance_feature = "feature" if "feature" in feature_importance.columns else feature_importance.columns[0]
            importance_value = "importance" if "importance" in feature_importance.columns else feature_importance.columns[-1]
            importance_view = feature_importance.head(15).copy()
            importance_view[importance_feature] = importance_view[importance_feature].map(_friendly_name)
            st.bar_chart(importance_view.set_index(importance_feature)[importance_value])

        with st.expander("Metrik lengkap"):
            st.json(metrics)
        st.markdown(
            """
            **Batasan penting**

            - Model dilatih untuk membedakan label historis *Dropout* dan *Graduate*;
              baris `Enrolled` tidak dipakai sebagai outcome final.
            - Fitur semester 2 dikeluarkan agar model dapat digunakan setelah semester 1.
            - Kode kategori belum dapat ditafsirkan tanpa data dictionary dari sumber dataset.
            - Informasi sensitif atau proksinya tidak boleh menjadi dasar tindakan merugikan;
              lakukan audit fairness sebelum penggunaan di lingkungan nyata.
            """
        )

    st.divider()
    st.caption("Educational prototype · skor risiko mendukung keputusan manusia, bukan menggantikannya.")


if __name__ == "__main__":
    main()
