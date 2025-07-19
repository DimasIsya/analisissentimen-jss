import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import emoji
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(layout="wide", page_title="Dashboard Analisis Sentimen JSS")

# KELAS INI HANYA DIPERLUKAN UNTUK DEMO PREPROCESSING DI TAB 2
# @st.cache_resource agar Sastrawi hanya di-load sekali
@st.cache_resource
def get_preprocessor():
    class TextPreprocessor:
        def __init__(self):
            self.stop_words = {'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'dalam', 'atau', 'ini', 'itu', 'juga', 'akan', 'sudah', 'telah', 'adalah', 'ada', 'pernah', 'sedang', 'masih', 'sering', 'selalu', 'kadang', 'biasanya', 'mungkin', 'segera', 'seperti', 'karena', 'kalau', 'jika', 'tapi', 'tetapi', 'namun', 'pas', 'serta', 'agar', 'supaya', 'oleh', 'sehingga', 'tersebut', 'yakni', 'yaitu', 'pula', 'saya', 'kamu', 'dia', 'mereka', 'kami', 'kita', 'ku', 'mu', 'pun', 'kah', 'lah', 'harus', 'buka', 'daftar', 'tolong', 'login', 'masuk', 'pilih', 'kirim', 'isi', 'pakai', 'akses', 'lihat', 'liat', 'buat', 'mohon', 'banget', 'sekali', 'aja', 'saja', 'saat', 'waktu', 'aplikasi', 'program', 'sistem', 'website', 'fitur', 'menu', 'akun', 'data', 'layanan', 'lapor', 'aduan', 'keluhan', 'informasi', 'info', 'verifikasi', 'update', 'jss', 'yogyakarta', 'masyarakat', 'min', 'admin', 'notifikasi', 'notif', 'email', 'upload', 'file', 'password', 'username', 'nomor', 'telepon', 'otp', 'nik', 'saran', 'solusi', 'bintang', 'wae', 'dab', 'lurr', 'sih', 'dong', 'deh', 'kok', 'koq', 'mah', 'nya', 'lho', 'toh', 'gais', 'kak', 'gan', 'bro', 'halo', 'hallo', 'hai', 'wah', 'apa', 'kenapa', 'bagaimana', 'kapan', 'dimana', 'kpd', 'dll', 'dsb', 'yth', 'cctv', 'tv', 'hotspot','maps','chat','tema','ndes','nggih'}
            self.slang_word_dict = {'eroorrr': 'error', 'eror': 'error', 'blom': 'belum', 'blm': 'belum', 'durung': 'belum', 'ndak': 'tidak', 'ga': 'tidak', 'gak': 'tidak', 'nggak': 'tidak', 'tdk': 'tidak', 'g': 'tidak', 'gk': 'tidak', 'ra': 'tidak', 'mboten': 'tidak', 'gaada': 'tidak ada', 'gabisa': 'tidak bisa', 'gakbisa': 'tidak bisa', 'bgt': 'banget', 'dgn': 'dengan', 'dg': 'dengan', 'krn': 'karena', 'utk': 'untuk', 'sdh': 'sudah', 'udh': 'sudah', 'dah': 'sudah', 'wes': 'sudah', 'sy': 'saya', 'dr': 'dari', 'seko': 'dari', 'klo': 'kalau', 'kalo': 'kalau', 'skrg': 'sekarang', 'lg': 'lagi', 'bnyk': 'banyak', 'byk': 'banyak', 'jg': 'juga', 'bkn': 'bukan', 'td': 'tadi', 'trs': 'terus', 'trus': 'terus', 'gmn': 'bagaimana', 'gimana': 'bagaimana', 'pripun': 'bagaimana', 'kpn': 'kapan', 'tgl': 'tanggal', 'thn': 'tahun', 'bln': 'bulan', 'kpd': 'kepada', 'ttp': 'tetap', 'sbg': 'sebagai', 'pdhl': 'padahal', 'great': 'bagus', 'bbrp': 'beberapa', 'hp': 'ponsel', 'tp': 'tapi', 'gpp': 'tidak apa-apa', 'sm': 'sama', 'lgsg': 'langsung', 'muter2': 'lambat', 'cmn': 'cuma', 'cuman': 'hanya', 'app': 'aplikasi', 'apps': 'aplikasi', 'knp': 'kenapa', 'msh': 'masih', 'jozz': 'bagus', 'tbtb': 'tiba-tiba', 'moga2': 'semoga', 'mhin': 'mohon', 'nomer': 'nomor', 'pasword': 'password', 'tlp': 'telepon', 'responnya': 'respons', 'lemot': 'lambat', 'ngelag': 'lambat', 'mubeng': 'lambat', 'muter-muter': 'lambat', 'muter': 'lambat', 'crash': 'error', 'force close': 'error tutup paksa', 'fc': 'error tutup paksa', 'loading': 'memuat', 'instal': 'pasang', 'uninstal': 'copot pemasangan', 'register': 'daftar', 'upgrade': 'tingkatkan', 'pemkot': 'pemerintah kota', 'pemda': 'pemerintah daerah', 'dindukcapil': 'dinas kependudukan pencatatan sipil', 'capil': 'pencatatan sipil', 'ktp': 'kartu tanda penduduk', 'kk': 'kartu keluarga', 'akte': 'akta', 'npwp': 'nomor pokok wajib pajak', 'opd': 'organisasi perangkat daerah', 'jogja': 'yogyakarta', 'yogya': 'yogyakarta', 'dishub': 'dinas perhubungan', 'dinkes': 'dinas kesehatan', 'suwun': 'terima kasih', 'okokok': 'oke', 'matur nuwun': 'terima kasih', 'maturnuwun': 'terima kasih', 'nuwun': 'terima kasih', 'ngeten': 'begini', 'lelet': 'lambat', 'angel': 'sulit', 'mumet': 'pusing', 'kesuwen': 'terlalu lama', 'suwe': 'lama', 'jan': 'sungguh', 'tenan': 'sungguh', 'malah': 'justru', 'iki': 'ini', 'opo': 'apa', 'iso': 'bisa', 'gawe': 'buat', 'ndelok': 'lihat', 'mantap': 'bagus', 'keren': 'bagus', 'good': 'bagus', 'nice': 'bagus', 'ok': 'oke', 'oke': 'oke', 'jos': 'bagus', 'joss': 'bagus', 'josss': 'bagus', 'mantab': 'bagus', 'mantuulll': 'bagus', 'mantabbb': 'bagus', 'apik': 'bagus', 'sae': 'bagus', 'istimewaa': 'bagus', 'sip': 'bagus', 'siip': 'bagus', 'sippppppppppp': 'bagus', 'tjakep': 'bagus', 'mantul': 'bagus', 'istimewa': 'bagus', 'top': 'bagus', 'the best': 'terbaik', 'best': 'terbaik', 'dabest': 'terbaik', 'luarbiasa': 'luar biasa', 'force close': 'error', 'bermanfaat': 'manfaat', 'membantu': 'bantu', 'helpful': 'bantu', 'sangat bantu': 'bantu', 'sangat membantu': 'bantu', 'susah': 'sulit', 'dpersulit': 'sulit', 'ribet': 'sulit', 'bertele-tele': 'sulit', 'berbelit': 'sulit', 'payah': 'buruk', 'ancur': 'hancur', 'parah': 'sangat buruk', 'elek': 'jelek', 'bosok': 'busuk', 'gajelas': 'tidak jelas', 'ngak': 'tidak', 'ga jelas': 'tidak jelas', 'zonk': 'gagal', 'thanks': 'terima kasih', 'makasih': 'terima kasih', 'mksh': 'terima kasih', 'trims': 'terima kasih', 'tks': 'terima kasih', 'trimakasih': 'terima kasih', 'diperbaiki': 'perbaiki', 'ditingkatkan': 'tingkat', 'ookkee' : 'oke', 'oke oke': 'oke', 'marai': 'bikin', 'mubeng minger': 'lambat','pekok': 'bodoh','kliteh': 'kriminalitas', 'rong': 'belum', 'nyobo': 'coba','aing': 'saya','bolan-baleni': 'berulang kali','nglebokke': 'memasukkan','bagussss': 'bagus', 'jempoooooooollllll': 'bagus', 'jempooll': 'bagus', 'verygood': 'sangat bagus', 'goodluck': 'semoga berhasil', "klk": "kalo", 'quick respon': 'respons cepat', 'optimal': 'maksimal', 'baguss': 'bagus', "joshh": 'bagus', "dadi": "jadi", 'b aja': 'biasa saja'}
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
        def preprocess_text(self, text):
            if pd.isna(text): return ""
            text = str(text).lower()
            text = re.sub(r'(.)\1{2,}', r'\1\1', text)
            for emoji_char, meaning in {'👍': ' bagus ', '❤️': ' suka ', '😊': ' senang ', '👌': ' oke ', '✅': ' setuju ', '✨': ' bagus sekali ', '⭐': ' bintang ', '💯': ' sempurna ', '👏': ' bagus ', '🙏': ' terima kasih ', '👎': ' jelek ', '😡': ' marah ', '😠': ' marah ', '😤': ' kesal ', '😭': ' kecewa menangis ', '😢': ' sedih ', '😞': ' kecewa ', '💔': ' kecewa ', '💩': ' jelek sekali ', '❌': ' salah '}.items():
                text = text.replace(emoji_char, meaning)
            text = emoji.demojize(text, delimiters=(" ", " ")).replace("_", " ")
            words = text.split()
            normalized_words = [self.slang_word_dict.get(word, word) for word in words]
            text = ' '.join(normalized_words)
            text = self.stemmer.stem(text)
            text = re.sub(r'http[s]?://\S+|www\.\S+', ' ', text)
            text = re.sub(r'[^a-zA-Z\s_]', ' ', text)
            text = ' '.join(text.split())
            negation_words = {'tidak', 'bukan', 'jangan', 'kurang', 'belum'}; words = text.split(); processed_words = []; i = 0
            while i < len(words):
                if words[i] in negation_words and i + 1 < len(words):
                    processed_words.append(words[i] + '_' + words[i+1]); i += 2
                else:
                    processed_words.append(words[i]); i += 1
            text = ' '.join(processed_words)
            words = text.split()
            words = [word for word in words if word not in self.stop_words]
            return ' '.join(words)
    return TextPreprocessor()

@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load('jss_sentiment_pipeline.pkl')
        return pipeline
    except FileNotFoundError:
        return None

@st.cache_data
def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return None

@st.cache_data
def load_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None

@st.cache_data
def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# =============================================================================
# Tampilan Utama Aplikasi
# =============================================================================
model_pipeline = load_model()
preprocessor = get_preprocessor()

st.title("🚀 Dashboard Analisis Sentimen Ulasan JSS")

if model_pipeline is None:
    st.error("❌ File model `jss_sentiment_pipeline.pkl` tidak ditemukan. Pastikan Anda sudah menjalankan skrip training terlebih dahulu.")
else:
    tab1, tab2 = st.tabs(["🔬 Proses Pembuatan Model", "▶️ Analisis Langsung"])

    with tab1:
        st.header("Kisah di Balik Model")
        
        st.subheader("1. Komposisi & Pembagian Data")
        split_info = load_json('output_split_info.json')
        sentiment_dist = load_csv('output_sentiment_distribution.csv')
        if split_info and sentiment_dist is not None:
            st.write(f"Model dilatih menggunakan **{split_info['total_count']} ulasan**. Data dibagi menjadi **{split_info['train_count']} data latih (75%)** dan **{split_info['test_count']} data uji (25%)**.")
            fig = px.pie(sentiment_dist, values='count', names='sentimen', title='Distribusi Sentimen pada Keseluruhan Data', color_discrete_map={'positif':'#66b3ff', 'negatif':'#ff9999', 'netral':'#99ff99'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("File distribusi sentimen atau info split tidak ditemukan.")
            
        with st.expander("Lihat 2. Tahapan Preprocessing & 3. Proses Pelabelan Data"):
            preprocessing_steps = load_csv('output_preprocessing_steps.csv')
            if preprocessing_steps is not None:
                st.dataframe(preprocessing_steps)
            else:
                st.warning("File langkah preprocessing tidak ditemukan.")
        
        with st.expander("Lihat 5. Hasil Pencarian Parameter Terbaik (Grid Search)"):
            gs_results = load_csv('output_grid_search_results.csv')
            if gs_results is not None:
                st.dataframe(gs_results[['param_classifier__C', 'param_tfidf__min_df', 'param_tfidf__ngram_range', 'mean_test_score', 'rank_test_score']].head())
            else:
                st.warning("File hasil Grid Search tidak ditemukan.")
        
        with st.expander("Lihat 6. Kata Kunci Utama per Sentimen"):
            top_features = load_json('output_top_features.json')
            if top_features:
                df_features = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in top_features.items() ])).head(10)
                st.dataframe(df_features, use_container_width=True)
            else:
                st.warning("File kata kunci utama tidak ditemukan.")
        
        st.subheader("8. Hasil Akhir: Akurasi & Confusion Matrix")
        report_str = load_text('output_classification_report.txt')
        if report_str:
            st.code(report_str, language='text')
        else:
            st.warning("File laporan klasifikasi tidak ditemukan.")
        
        if os.path.exists('output_confusion_matrix.png'):
            st.image("output_confusion_matrix.png", caption="Confusion Matrix pada data uji.")
        else:
            st.warning("File gambar confusion matrix tidak ditemukan.")

    with tab2:
        st.header("Analisis Sentimen Langsung")
        user_input = st.text_area("Tulis ulasan Anda di sini:", "Aplikasinya keren dan sangat membantu, tapi kadang masih suka error.", height=130)
        
        if st.button("Analisis Sekarang", type="primary", use_container_width=True):
            if user_input:
                st.markdown("---")
                st.subheader("Hasil Analisis")
                
                st.markdown("##### Langkah 1: Proses Pembersihan Teks")
                final_text_for_model = preprocessor.preprocess_text(user_input)
                with st.expander("Lihat Teks Bersih (Siap Olah)"):
                    st.code(final_text_for_model, language="text")
                
                # =============================================================================
                st.markdown("##### Langkah 2: Skor Bobot Kata (TF-IDF)")
                st.info("Ini adalah skor TF-IDF **final** dari kata-kata dalam teks, yang menunjukkan seberapa penting kata tersebut bagi model setelah melalui semua perhitungan termasuk normalisasi.")
                
                vectorizer = model_pipeline.named_steps['tfidf']
                input_vector = vectorizer.transform([final_text_for_model])
                feature_names = vectorizer.get_feature_names_out()
                
                feature_scores = []
                # Iterasi hanya pada kata-kata yang ada di input Anda
                for col_index in input_vector.nonzero()[1]:
                    # Ambil nama kata dan skor TF-IDF finalnya
                    word = feature_names[col_index]
                    score = input_vector[0, col_index]
                    feature_scores.append((word, score))
                
                if feature_scores:
                    df_scores = pd.DataFrame(feature_scores, columns=['Kata', 'Skor TF-IDF Final']).sort_values(by='Skor TF-IDF Final', ascending=False)
                    st.dataframe(df_scores.style.format({'Skor TF-IDF Final': '{:.4f}'}), use_container_width=True)
                else:
                    st.warning("Tidak ada kata yang dikenali oleh model dari teks Anda.")

                st.markdown("##### Langkah 3: Prediksi & Tingkat Kepercayaan")
                prediction = model_pipeline.predict([final_text_for_model])[0]
                probability = model_pipeline.predict_proba([final_text_for_model])
                
                if prediction == 'positif': st.success(f"**Hasil Prediksi: POSITIF** 👍")
                elif prediction == 'negatif': st.error(f"**Hasil Prediksi: NEGATIF** 👎")
                else: st.warning(f"**Hasil Prediksi: NETRAL** 😐")
                
                prob_df = pd.DataFrame(probability, columns=model_pipeline.classes_, index=["Tingkat Kepercayaan"]).T
                prob_df = prob_df.sort_values(by="Tingkat Kepercayaan", ascending=False)
                st.write("Skor Kepercayaan Model:"); st.dataframe(prob_df.style.format("{:.2%}")); st.bar_chart(prob_df)
            else:
                st.warning("Mohon masukkan teks terlebih dahulu untuk dianalisis.")