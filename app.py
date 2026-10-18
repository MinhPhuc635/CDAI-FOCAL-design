from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import streamlit as st

import focal_pipeline as focal


ROOT = Path(__file__).parent
DATA_DIR = Path(os.getenv("MOD_VEHICLE_DATA_DIR", r"F:\MOD_vehicle\raw_data"))
MODEL_PATH = focal.MODEL_PATH
LABELS_PATH = focal.LABELS_PATH
METRICS_PATH = focal.METRICS_PATH

st.set_page_config(page_title="Vehicle Acoustic–Seismic Classifier", page_icon="◈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
:root{--blue:#3569ce;--ink:#20344f;--muted:#71839a;--line:#dce5f0;--pale:#eef4fb;--green:#27976b}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
.stApp{background:#f6f8fb}.block-container{max-width:1500px;padding-top:1rem;padding-bottom:2rem}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:8px 10px 20px;border-bottom:1px solid #e1e8f1;margin-bottom:24px}
.brand{font-size:23px;font-weight:800;letter-spacing:-.5px}.brand b{color:var(--blue)}.brand-mark{display:inline-flex;width:43px;height:43px;border:2px solid #e4edf9;border-radius:50%;align-items:center;justify-content:center;color:var(--blue);margin-right:12px;font-size:22px}
.user{color:var(--muted);font-size:13px}.hero{padding:0 6px 18px}.eyebrow{font-size:11px;font-weight:800;letter-spacing:1.7px;color:var(--blue);text-transform:uppercase}.title{font-size:28px;letter-spacing:-.8px;font-weight:800;margin:3px 0}.subtitle{font-size:13px;color:var(--muted)}
[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border:1px solid var(--line);border-radius:18px;box-shadow:0 3px 12px #1d38521a;padding:18px 18px 15px}
.section-title{font-size:18px;font-weight:800;margin:2px 0 16px}.icon{display:inline-flex;background:#edf3fb;color:var(--blue);height:42px;width:42px;border-radius:12px;align-items:center;justify-content:center;margin-right:10px;font-size:20px;vertical-align:middle}
.drop-note{border:1px dashed #b9cce6;border-radius:14px;background:#f8fafd;padding:30px 14px;text-align:center;color:var(--muted);margin:10px 0 15px}.drop-note strong{display:block;color:var(--ink);font-size:14px;margin:7px 0 2px}.statusbox{background:#eff7f3;border-radius:14px;padding:16px}.pill{display:inline-block;background:#dff1e8;color:#237b58;border-radius:10px;padding:9px 12px;font-weight:700;width:100%;font-size:13px}.resultbox{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px;margin:8px 0 14px}.pred{font-size:27px;font-weight:800;color:var(--blue);margin:2px 0}.conf{font-size:36px;font-weight:800;color:var(--blue)}.muted{color:var(--muted);font-size:12px}.metric{font-size:25px;font-weight:800;color:var(--blue)}
div.stButton>button{height:48px;border-radius:11px;font-weight:700;border:1px solid var(--blue);color:var(--blue)}div.stButton>button[kind="primary"]{background:var(--blue);color:white}.stProgress>div>div>div>div{background:var(--green)}
@media(max-width:850px){.title{font-size:23px}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return focal.load_focal_model("cpu")


def load_labels(path: Path):
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_matches_dataset() -> bool:
    if not (MODEL_PATH.exists() and LABELS_PATH.exists() and METRICS_PATH.exists()):
        return False
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metrics.get("dataset_id") == "MOD_vehicle"


def inspect_raw_uploads(files) -> tuple[list[dict], list, list]:
    """Describe uploads and find likely acoustic and seismic input files."""
    info = []
    audio_candidates = []
    seismic_candidates = []
    for file in files:
        item = {"Tệp": file.name, "Kích thước": f"{file.size / (1024 * 1024):.1f} MB", "Định dạng": Path(file.name).suffix.lower()}
        suffix = Path(file.name).suffix.lower()
        name = Path(file.name).name.lower()
        if suffix == ".wav":
            audio_candidates.append(file)
            try:
                file.seek(0)
                with wave.open(file, "rb") as audio:
                    item["Thông tin"] = f"{audio.getframerate():,} Hz · {audio.getnchannels()} kênh · {audio.getnframes() / audio.getframerate():.1f} giây"
            except (wave.Error, EOFError):
                item["Thông tin"] = "WAV không đọc được"
        elif suffix == ".csv" and name.startswith(("aud", "audio")):
            audio_candidates.append(file)
            item["Thông tin"] = "Âm thanh · CSV theo cấu trúc MOD"
        elif suffix == ".csv":
            seismic_candidates.append(file)
            if name == "ehz.csv":
                item["Thông tin"] = "Địa chấn dọc EHZ · CSV MOD"
            else:
                item["Thông tin"] = "CSV địa chấn hoặc dữ liệu phụ · kiểm tra đúng thành phần EHZ"
        else:
            item["Thông tin"] = "Định dạng chưa được hỗ trợ"
        info.append(item)
    return info, audio_candidates, seismic_candidates


def ensure_checkpoint() -> None:
    if checkpoint_matches_dataset():
        return
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Không thấy dataset MOD_vehicle tại {DATA_DIR}. Cấu hình MOD_VEHICLE_DATA_DIR trên máy chủ.")
    env = os.environ.copy()
    env.setdefault("MOD_VEHICLE_DATA_DIR", str(DATA_DIR))
    log_path = ROOT / "models" / "training.log"
    with st.status("Chưa có checkpoint. Đang tự tiền xử lý dữ liệu và huấn luyện FOCAL lần đầu…", expanded=True) as status:
        st.write("Bước 1/2 · Tạo cửa sổ tín hiệu và chia tập huấn luyện/kiểm tra.")
        st.write("Bước 2/2 · Tiền huấn luyện FOCAL, fine-tune bộ phân loại và đo accuracy trên tập giữ lại.")
        process = subprocess.run(
            [sys.executable, str(ROOT / "train_focal.py")],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=False,
        )
        if process.returncode != 0 or not checkpoint_matches_dataset():
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
            except OSError:
                tail = ""
            status.update(label="Tạo checkpoint thất bại", state="error", expanded=True)
            st.error("Quá trình train không hoàn tất. Nhật ký cuối:")
            st.code(tail or f"Chi tiết: {log_path}")
            raise RuntimeError("FOCAL checkpoint training failed")
        status.update(label="FOCAL checkpoint đã sẵn sàng", state="complete", expanded=False)


st.markdown('<div class="topbar"><div class="brand"><span class="brand-mark">◈</span>Vehicle <b>Acoustic–Seismic</b></div><div class="user">FOCAL · MOD_vehicle dataset&nbsp;&nbsp; ●</div></div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="eyebrow">Multimodal sensing intelligence</div><div class="title">Vehicle Acoustic–Seismic Classifier</div><div class="subtitle">Nhận diện phương tiện từ tín hiệu âm thanh và địa chấn bằng mô hình FOCAL.</div></div>', unsafe_allow_html=True)

model_ready = checkpoint_matches_dataset()
labels = load_labels(LABELS_PATH)
col_a, col_b, col_c = st.columns([1.06, 1.02, 1.06], gap="medium")

with col_a:
    with st.container(border=True):
        st.markdown('<div class="section-title"><span class="icon">⇧</span>Dữ liệu đầu vào</div>', unsafe_allow_html=True)
        st.markdown('<div class="drop-note"><span style="font-size:34px;color:#80a7dc">☁</span><strong>Tải tín hiệu MOD gốc</strong>.csv · .wav</div>', unsafe_allow_html=True)
        uploads = st.file_uploader("Chọn các file tín hiệu", type=["csv", "wav"], accept_multiple_files=True, label_visibility="collapsed")
        audio_file = None
        seismic_file = None
        upload_info = []
        if uploads:
            upload_info, audio_candidates, seismic_candidates = inspect_raw_uploads(uploads)
            st.dataframe(upload_info, hide_index=True, use_container_width=True)
            if len(audio_candidates) == 1:
                audio_file = audio_candidates[0]
            elif audio_candidates:
                audio_choices = {f"{index + 1}. {f.name} ({f.size / (1024 * 1024):.1f} MB)": f for index, f in enumerate(audio_candidates)}
                selected_audio = st.selectbox("Chọn file âm thanh", list(audio_choices), key="audio_upload")
                audio_file = audio_choices[selected_audio]

            ehz_candidates = [f for f in seismic_candidates if Path(f.name).name.lower() == "ehz.csv"]
            if len(ehz_candidates) == 1:
                seismic_file = ehz_candidates[0]
            elif seismic_candidates:
                st.warning("Không thấy file tên ehz.csv. Hãy chọn đúng CSV thành phần địa chấn dọc EHZ.")
                seismic_choices = {f"{index + 1}. {f.name} ({f.size / (1024 * 1024):.1f} MB)": f for index, f in enumerate(seismic_candidates)}
                selected_seismic = st.selectbox("Chọn CSV địa chấn EHZ", list(seismic_choices), key="seismic_upload")
                seismic_file = seismic_choices[selected_seismic]

            if audio_file and seismic_file:
                st.success(f"Đã chọn âm thanh `{audio_file.name}` và địa chấn `{seismic_file.name}`.")
            else:
                st.info("FOCAL cần cả âm thanh (.wav/aud*.csv) và địa chấn EHZ (.csv) đồng bộ. Nút dự đoán vẫn bấm được để xem hướng dẫn thiếu file.")
with col_b:
    with st.container(border=True):
        st.markdown('<div class="section-title"><span class="icon">▣</span>FOCAL Model</div>', unsafe_allow_html=True)
        st.markdown('<div class="statusbox"><div style="font-weight:700;margin-bottom:12px">Trạng thái mô hình</div>', unsafe_allow_html=True)
        if model_ready:
            st.markdown('<div class="pill">● &nbsp;Checkpoint đã sẵn sàng</div>', unsafe_allow_html=True)
            st.progress(1.0, text="FOCAL fine-tuned · sẵn sàng suy luận")
        else:
            st.info("Checkpoint MOD_vehicle chưa sẵn sàng. Khi bấm chạy dự đoán, hệ thống sẽ train FOCAL từ MOD_VEHICLE_DATA_DIR một lần rồi lưu lại để dùng cho các lần sau.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")
        run = st.button("▶  Chạy dự đoán", type="primary", use_container_width=True)
        if run:
            if not uploads:
                st.warning("Trước tiên hãy tải lên file âm thanh và file CSV địa chấn EHZ.")
            elif not audio_file or not seismic_file:
                missing = []
                if not audio_file:
                    missing.append("âm thanh (.wav hoặc aud*.csv)")
                if not seismic_file:
                    missing.append("địa chấn EHZ (.csv, thường tên ehz.csv)")
                st.warning("Còn thiếu: " + " và ".join(missing) + ".")
            else:
                signature = tuple(sorted((file.name, file.size) for file in uploads))
                try:
                    ensure_checkpoint()
                    if Path(audio_file.name).suffix.lower() == ".wav":
                        audio = focal.read_audio_wav(audio_file)
                    else:
                        audio = focal.read_audio_csv(audio_file)
                    seismic = focal.read_seismic_csv(seismic_file)
                    audio_windows, seismic_windows = focal.make_time_windows(audio, seismic)
                    model, loaded_labels = load_model(str(MODEL_PATH))
                    probabilities = focal.predict_windows(model, audio_windows, seismic_windows)
                    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
                    st.session_state["prediction"] = {
                        "labels": loaded_labels,
                        "probs": probabilities.tolist(),
                        "test_accuracy_percent": metrics["test_accuracy_percent"],
                        "independent_recording_accuracy_percent": metrics.get("test_independent_recording_accuracy_percent"),
                        "independent_recordings": metrics.get("test_independent_recordings"),
                        "signature": signature,
                        "windows": len(audio_windows),
                    }
                except Exception as exc:
                    st.error(f"Không thể hoàn tất dự đoán: {exc}")

with col_c:
    with st.container(border=True):
        st.markdown('<div class="section-title"><span class="icon">▥</span>Kết quả</div>', unsafe_allow_html=True)
        result = st.session_state.get("prediction")
        current_signature = tuple(sorted((file.name, file.size) for file in uploads)) if uploads else None
        if result and result.get("signature") != current_signature:
            result = None
        if result:
            order = np.argsort(result["probs"])[::-1]
            idx = int(order[0]); confidence = result["probs"][idx] * 100
            recording_score = result.get("independent_recording_accuracy_percent")
            accuracy_text = f'Test accuracy theo cửa sổ · {result["test_accuracy_percent"]:.1f}%'
            if recording_score is not None:
                accuracy_text += f' · bản ghi độc lập · {recording_score:.1f}% ({result["independent_recordings"]} lượt)'
            st.markdown(f'<div class="resultbox"><div class="muted">Phương tiện nhận diện</div><div class="pred">{result["labels"][idx]}</div><div class="muted">Độ tin cậy dự đoán</div><div class="conf">{confidence:.1f}%</div><div class="muted">{accuracy_text}</div></div>', unsafe_allow_html=True)
            st.caption(f"Tổng hợp từ {result['windows']} cửa sổ tín hiệu, mỗi cửa sổ 2 giây.")
            st.markdown("**Xác suất các lớp khác**")
            for i in order[1:6]:
                st.progress(float(result["probs"][i]), text=f"{result['labels'][i]} · {result['probs'][i]*100:.1f}%")
        elif not model_ready:
            st.markdown('<div class="resultbox"><div class="muted">Phân loại phương tiện</div><div class="pred">FOCAL sẵn sàng tự train</div><div class="muted">Checkpoint sẽ được tạo tự động khi chạy dự đoán lần đầu.</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="resultbox"><div class="muted">Phân loại phương tiện</div><div class="pred">Chưa có kết quả</div><div class="muted">Tải tensor đầu vào rồi chạy dự đoán.</div></div>', unsafe_allow_html=True)
