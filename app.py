from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy # type: ignore
from flask_session import Session
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from zoneinfo import ZoneInfo, available_timezones
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import http.client
import json

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = 'supersecretkey'  # Add secret key for session
RAPIDAPI_KEY = '52b8c0013dmshe8c44c191967aaep1bbde0jsne21ac58f675c'
RAPIDAPI_HOST = 'chatgpt-42.p.rapidapi.com'
RAPIDAPI_URL = '/matag2'

db = SQLAlchemy(app)
Session(app)


# Load Model
model = load_model('earthquake_modellllllll.keras')

# Provinces used during model training
provinces = sorted([
    'Nusa Tenggara Timur', 'Papua', 'Maluku', 'Gorontalo', 'Riau', 'Sumatra Utara',
    'Nusa Tenggara Barat', 'Yogyakarta', 'Sulawesi Utara', 'Maluku Utara',
    'Banten', 'Jawa Barat', 'Jawa Timur', 'Aceh', 'Sulawesi Tengah',
    'Sumatra Barat', 'Bali', 'Papua Barat', 'Sumatra Selatan', 'Lampung',
    'Jawa Tengah', 'Kalimantan Timur', 'Bengkulu', 'Sulawesi Tenggara', 'Jambi',
    'Sulawesi Barat', 'Kalimantan Barat', 'Sulawesi Selatan', 'Kalimantan Selatan', 'Kalimantan Utara'
])

# Encoder used during model training
le = LabelEncoder()
le.fit(provinces)  # Fit encoder with complete list of provinces

def preprocess_input(provinsi, kedalaman, magnitudo):
    # Encode province
    provinsi_encoded = le.transform([provinsi])
    # No normalization on depth and magnitude
    return np.array([provinsi_encoded[0], kedalaman, magnitudo]).reshape(1, -1)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        month = int(request.form.get('month'))
        year = int(request.form.get('year'))
    else:
        # Default to August 2024
        month = 8
        year = 2024

    if 'selected_month' in session and 'selected_year' in session and \
       session['selected_month'] == month and session['selected_year'] == year:
        # If selected month and year are the same as in session, retrieve prediction from session
        df = pd.read_json(session.get('df'))
    else:
        # Convert month and year to datetime using pandas
        start_date = pd.Timestamp(year=year, month=month, day=1)
        end_date = start_date + pd.DateOffset(months=1) - pd.DateOffset(days=1)

        # Generate date range for prediction
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        predictions = []
        for date in date_range:
            province = np.random.choice(provinces)  # Choose a random province from the list
            kedalaman = np.random.uniform(10, 150)  # Use raw values for depth
            magnitudo = np.random.uniform(4, 6)     # Use raw values for magnitude
            data = preprocess_input(province, kedalaman, magnitudo)
            pred = model.predict(data)
            predictions.append([date.strftime('%Y-%m-%d'), pred[0][0], pred[0][1], province])

        df = pd.DataFrame(predictions, columns=['tanggal', 'pred_depth', 'pred_mag', 'provinsi'])

        # Save predictions in session
        session['df'] = df.to_json()
        session['selected_month'] = month
        session['selected_year'] = year

    # Prepare data for chart
    chart_data = {
        'labels': df['tanggal'].tolist(),
        'depth': df['pred_depth'].tolist(),
        'magnitude': df['pred_mag'].tolist()
    }
    
    # Fetch recent articles
    articles = Article.query.order_by(Article.date_posted.desc()).limit(4).all()

    return render_template(
        'index2.html',
        tables=[df.to_html(classes='data')],
        chart_data=chart_data,
        selected_month=month,
        selected_year=year,
        provinces=provinces,
        selected_province=session.get('selected_province', ''),
        articles=articles  # Pass articles to the template
    )

@app.route('/search', methods=['POST'])
def search():
    province = request.form.get('province')
    
    # Retrieve month and year from session
    month = session.get('selected_month')
    year = session.get('selected_year')
    
    # Retrieve predictions from session
    df = pd.read_json(session.get('df'))

    # Filter data by selected province
    filtered_df = df[df['provinsi'] == province]

    if filtered_df.empty:
        flash(f"Tidak ada data untuk provinsi {province}", 'warning')
        session['search_empty'] = True
        chart_data = {'labels': [], 'depth': [], 'magnitude': []}
        tables = []
    else:
        session.pop('search_empty', None)
        chart_data = {
            'labels': filtered_df['tanggal'].tolist(),
            'depth': filtered_df['pred_depth'].tolist(),
            'magnitude': filtered_df['pred_mag'].tolist()
        }
        tables = [filtered_df.to_html(classes='data')]

    session['selected_province'] = province

    # Fetch recent articles
    articles = Article.query.order_by(Article.date_posted.desc()).limit(4).all()

    return render_template(
        'index2.html',
        tables=tables,
        chart_data=chart_data,
        selected_month=month,
        selected_year=year,
        provinces=provinces,
        selected_province=province,
        articles=articles
    )


    
#KLASIFIKASI
model1 = load_model('2-Model CNN.h5')

dataset1 = pd.read_csv('tektonik.csv')
dataset1.drop(columns=['date'], inplace=True)
label_encoder1 = LabelEncoder()
dataset1['kelas'] = label_encoder1.fit_transform(dataset1['kelas'])

class logVulkanik(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    amplitudo = db.Column(db.Float, nullable=False)
    durasi = db.Column(db.Float, nullable=False)
    classification_result = db.Column(db.String(100), nullable=False)

class logTektonik(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    kedalaman = db.Column(db.Float, nullable=False)
    magnitudo = db.Column(db.Float, nullable=False)
    classification_result = db.Column(db.String(100), nullable=False)

# Function to get earthquake classification label for the first model
def get_classification_label1(prediction):
    if prediction == 0:
        return "DALAM"
    elif prediction == 1:
        return "DANGKAL"
    elif prediction == 2:
        return "MENENGAH"

# Function to get earthquake classification text for the first model
def get_classification_text1(prediction):
    if prediction == 0:
        return "Gempa bumi dalam adalah gempa bumi yang hiposenternya berada lebih dari 300 km di bawah permukaan bumi. Di Indonesia gempa bumi ini berada di Laut Jawa, Laut Flores, Laut Banda dan Laut Sulawesi. Gempa ini tidak membahayakan."
    elif prediction == 1:
        return "Gempa bumi dangkal adalah gempa bumi yang hiposenternya berada kurang dari 50 km dari permukaan bumi. Di Indonesia gempa bumi dangkal letaknya terpencar di sepanjang sesar aktif dan patahan aktif. Gempa ini menimbulkan kerusakan besar dan semakin dangkal tempat terjadinya maka kerusakan semakin besar."
    elif prediction == 2:
        return "Gempa bumi menengah adalah gempa bumi yang hiposenternya berada antara 50 km–300 km di bawah permukaan bumi. Di Indonesia gempa bumi ini terbentang sepanjang Sumatra sebelah Barat, Jawa sebelah Selatan, selanjutnya Nusa Tenggara antara Sumbawa dan Maluku, sepanjang Teluk Tomini, dan Laut Maluku sampai Filipina. Gempa ini dengan focus kurang dari 150 km dibawah permukaan bumi masih dapat menimbulkan kerusakan."

@app.route('/klasifikasi_tektonik', methods=['GET', 'POST'])
def klasifikasi_tektonik():
    interpreter1 = tf.lite.Interpreter(model_path='tektonik.tflite')
    interpreter1.allocate_tensors()
    if request.method == 'POST':
        data = request.get_json()
        magnitudo = float(data['magnitudo'])
        kedalaman = float(data['kedalaman'])

        input_details = interpreter1.get_input_details()
        output_details = interpreter1.get_output_details()

        input_shape = input_details[0]['shape']
        print(f"Expected input shape: {input_shape}")

        input_data = np.array([[[magnitudo], [kedalaman]]], dtype=np.float32)

        # Reshape the input data to match the expected input shape
        input_data = np.reshape(input_data, input_shape)

        # Verify the new shape of the input data
        print(f"Input data shape: {input_data.shape}")

        interpreter1.set_tensor(input_details[0]['index'], input_data)

        # Run the inference
        interpreter1.invoke()

        # Extract the output
        predicted_class = interpreter1.get_tensor(output_details[0]['index'])

        predicted_class1 = np.argmax(predicted_class, axis=1)

        print(f"hasil prediksi: {predicted_class1}")

        # Make prediction
        # prediction_prob1 = model1.predict(np.array([[magnitudo, kedalaman]]))[0]
        # predicted_class1 = np.argmax(prediction_prob1)

        classification_label1 = get_classification_label1(predicted_class1)
        classification_text1 = get_classification_text1(predicted_class1)

        new_log = logTektonik(
            timestamp=datetime.now(ZoneInfo("Asia/Jakarta")),
            kedalaman=kedalaman,
            magnitudo=magnitudo,
            classification_result=classification_label1
        )
        db.session.add(new_log)
        db.session.commit()

        return jsonify({
            'label': classification_label1,
            'text': classification_text1
        })
    return render_template('klasifikasi_tektonik.html')

@app.route('/log_tektonik')
def log_tektonik():
    logs = logTektonik.query.order_by(logTektonik.timestamp.desc()).all()
    return render_template('log_tektonik.html', logs=logs)

# Load the second model and dataset
model2 = load_model('2-Model CNN_vulkanik.h5')

# Function to get earthquake classification label for the second model
def get_classification_label2(prediction):
    if prediction == 0:
        return "AWANPANAS (Pyroclastic Flow)"
    elif prediction == 1:
        return "LF (Low-Frequency)"
    elif prediction == 2:
        return "MP (Multi-Phase)"
    elif prediction == 3:
        return "ROCKFALL (Longsoran Batu)"
    elif prediction == 4:
        return "TECLOC (Tektonik Lokal)"
    elif prediction == 5:
        return "TECT (Tektonik)"
    elif prediction == 6:
        return "VTB (Vulkanotektonik Dalam)"

# Function to get earthquake classification text for the second model
def get_classification_text2(prediction):
    if prediction == 0:
        return "Gempa ini berhubungan dengan awan panas atau aliran piroklastik yang terjadi saat letusan gunung berapi. Awan panas adalah campuran gas panas, abu vulkanik, dan batuan yang bergerak cepat menuruni lereng gunung berapi, menghasilkan getaran yang dapat terdeteksi sebagai gempa."
    elif prediction == 1:
        return "Gempa Low-Frequency memiliki frekuensi getaran yang rendah dan biasanya diakibatkan oleh pergerakan fluida magma atau gas di dalam gunung berapi. Gempa ini sering dikaitkan dengan aktivitas vulkanik yang sedang berlangsung atau akan terjadi."
    elif prediction == 2:
        return "Gempa Multi-Phase adalah jenis gempa yang memiliki beberapa fase getaran, yang dihasilkan oleh kombinasi aktivitas vulkanik seperti pergerakan magma dan gas di bawah permukaan bumi. Gempa ini seringkali kompleks dan memiliki pola getaran yang bervariasi."
    elif prediction == 3:
        return "Gempa ini terjadi ketika batuan besar atau material lainnya jatuh dari lereng gunung berapi. Longsoran batu dapat terjadi karena gravitasi atau sebagai akibat dari letusan gunung berapi. Gempa jenis ini biasanya berfrekuensi tinggi dan berdurasi singkat."
    elif prediction == 4:
        return "Gempa Tektonik Lokal adalah gempa yang terjadi akibat aktivitas tektonik di sekitar daerah gunung berapi. Gempa ini memiliki karakteristik yang mirip dengan gempa tektonik umum tetapi terjadi di area yang lebih lokal, sering kali dipengaruhi oleh pergerakan magma atau tekanan di dalam kerak bumi di sekitar gunung berapi."
    elif prediction == 5:
        return "Gempa Tektonik adalah gempa yang terjadi karena pergerakan lempeng tektonik dan bukan langsung disebabkan oleh aktivitas vulkanik. Namun, gempa ini dapat mempengaruhi aktivitas gunung berapi jika terjadi di dekatnya atau memicu letusan vulkanik."
    elif prediction == 6:
        return "Gempa Vulkanotektonik Dalam terjadi pada kedalaman yang lebih besar di bawah gunung berapi, biasanya disebabkan oleh retakan atau pergerakan magma yang mempengaruhi struktur tektonik di dalam bumi. Gempa ini sering kali memiliki magnitudo yang lebih besar dibandingkan dengan gempa vulkanik lainnya."

@app.route('/klasifikasi_vulkanik', methods=['GET', 'POST'])
def klasifikasi_vulkanik():
    interpreter2 = tf.lite.Interpreter(model_path='vulkanik.tflite')
    interpreter2.allocate_tensors()
    if request.method == 'POST':
        data = request.get_json()
        Duration = float(data['Duration'])
        Amplitude = float(data['Amplitude'])


        input_details2 = interpreter2.get_input_details()
        output_details2 = interpreter2.get_output_details()

        input_shape2 = input_details2[0]['shape']
        print(f"Expected input shape: {input_shape2}")

        input_data2 = np.array([[[Duration], [Amplitude]]], dtype=np.float32)

        # Reshape the input data to match the expected input shape
        input_data2 = np.reshape(input_data2, input_shape2)

        # Verify the new shape of the input data
        print(f"Input data shape: {input_data2.shape}")

        interpreter2.set_tensor(input_details2[0]['index'], input_data2)

        # Run the inference
        interpreter2.invoke()

        # Extract the output
        prediction_prob2 = interpreter2.get_tensor(output_details2[0]['index'])
        
        # Make prediction
        # prediction_prob2 = model2.predict(np.array([[Duration, Amplitude]]))[0]
        predicted_class2 = np.argmax(prediction_prob2)

        classification_label2 = get_classification_label2(predicted_class2)
        classification_text2 = get_classification_text2(predicted_class2)

        new_log = logVulkanik(
            timestamp=datetime.now(ZoneInfo("Asia/Jakarta")),
            amplitudo=Amplitude,
            durasi=Duration,
            classification_result=classification_label2
        )
        db.session.add(new_log)
        db.session.commit()

        return jsonify({
            'label': classification_label2,
            'text': classification_text2
        })
    return render_template('klasifikasi_vulkanik.html')

@app.route('/log_vulkanik')
def log_vulkanik():
    logs = logVulkanik.query.order_by(logVulkanik.timestamp.desc()).all()
    return render_template('log_vulkanik.html', logs=logs)

# Load location data from CSV
df = pd.read_csv('chatbott.csv')
# Ensure latitude and longitude are in proper float format
df['Latitude'] = df['Latitude'].astype(str).str.replace(',', '').astype(float)
df['Longitude'] = df['Longitude'].astype(str).str.replace(',', '').astype(float)

@app.route('/chatbott')
def chatbott():
    return render_template('chatbott.html')

@app.route('/location/nearest', methods=['POST'])
def get_nearest_location():
    data = request.json
    print("Request JSON:", data)  # Debugging line
    user_lat = float(data['lat'])
    user_lon = float(data['lon'])
    
    # Calculate distance using Haversine formula
    df['distance'] = ((df['Latitude'] - user_lat)**2 + (df['Longitude'] - user_lon)**2)**0.5
    nearest_location = df.loc[df['distance'].idxmin()]
    
    return jsonify(nearest_location.to_dict())

# Model Article
class Article(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    image_url = db.Column(db.String(100), nullable=False)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# Fungsi untuk menambahkan artikel default ke database
def add_default_articles():
    articles = [
        {
            'title': 'Gempa M 7,2 Guncang Peru, Picu Peringatan Tsunami',
            'content': 'Gempa bumi dengan Magnitudo 7,2 mengguncang wilayah lepas pantai Peru pada Jumat (28/6) waktu setempat. Gempa kuat ini memicu peringatan tsunami yang diperkirakan akan menerjang sejumlah garis pantai negara Amerika Selatan tersebut. Seperti dilansir AFP, Jumat (28/6/2024), Survei Geologi Amerika Serikat (AS) atau USGS mencatat gempa bumi itu terjadi di perairan berjarak sekitar 8,8 kilometer dari distrik Atiquipa, Peru. Laporan USGS menyebut guncangan kuat akibat gempa dirasakan di area-area yang berada di sekitar pusat gempa. Belum ada laporan kerusakan atau korban jiwa akibat gempa bumi ini. Pusat Peringatan Tsunami Pasifik, secara terpisah, merevisi buletin yang sebelumnya menyatakan tidak ada ancaman akibat gempa di Peru tersebut. "Gelombang tsunami yang berbahaya diperkirakan terjadi di beberapa area pantai," demikian pernyataan Pusat Peringatan Tsunami Pasifik. Disebutkan bahwa gelombang tsunami tersebut bisa mencapai ketinggian antara "satu meter hingga tiga meter di atas permukaan air pasang". Peru yang merupakan negara dengan sekitar 33 juta jiwa penduduk, berada di area Cincin Api Pasifik yang merupakan area luas dengan aktivitas seismik intens yang membentang di sepanjang pantai barat Benua Amerika. Wilayah Peru dilanda ratusan gempa bumi yang terdeteksi setiap tahunnya.',
            'image_url': '/static/images/gambar1.jpg',
            'date_posted': datetime(2023, 6, 1)
        },
        {
            'title': 'Gempa M 3,2 Guncang Tuban, Berpusat di Darat',
            'content': 'Gempa bumi dengan kekuatan Magnitudo (M) 3,2 mengguncang Tuban, Jawa Timur (Jatim). Gempa tersebut berpusat di darat. Menurut data Badan Meteorologi, Klimatologi, dan Geofisika (BMKG), menyebut gempa tersebut terjadi pada Sabtu (29/6/2024), pukul 04.15 WIB. Lokasi gempa berada pada koordinat 6,94 lintang selatan, dan 111,83 bujur timur. "22 km barat daya Tubang, Jatim," tulis BMKG. BMKG menyebut pusat gempa berada pada kedalaman 10 km. Belum ada informasi mengenai ada tidaknya kerusakan dan korban akibat gempa tersebut. "Disclaimer, informasi ini mengutamakan kecepatan, sehingga hasil pengolahan data belum stabil dan bisa berubah seiring kelengkapan data," katanya.',
            'image_url': '/static/images/gambar2.jpg',
            'date_posted': datetime(2023, 6, 2)
        },
        {
            'title': 'Guncangan Tanah Akibat Gempa Garut 27 April 2024',
            'content': 'Telah terjadi gempabumi pada hari Sabtu tanggal 27 April 2024 jam 23:29:47 WIB dengan magnitude 6.2. Pusat Gempabumi (epicenter) terletak pada koordinat 8.39°LS 107.11°BT terletak di 156 km BaratDaya KAB-GARUT-JABAR pada kedalaman 70 km. Sumber gempabumi yang berada dilaut dengan kedalaman 70 km tersebut berasal dari zona Intraslab. Jenis patahan dan mekanisme sumber gempa yang terjadi pada gempa bumi berkekuatan magnitudo 6,5 di Garut, Jawa Barat adalah sesar naik (thrust fault). Hal ini didasarkan pada analisis mekanisme sumber yang menunjukkan bahwa gempa bumi tersebut memiliki mekanisme pergerakan naik (thrust fault). Jenis patahan di mana blok batuan di atas patahan bergerak ke atas relatif terhadap blok di bawahnya, yang merupakan karakteristik dari gempa bumi dengan mekanisme pergerakan naik.',
            'image_url': '/static/images/gambar3.jpg',
            'date_posted': datetime(2023, 6, 3)
        },
        {
            'title': 'BMKG Ungkap Kejadian Gempabumi Sumedang',
            'content': 'Sumedang, 07 Januari 2024 - Pasca gempabumi merusak M4,8 di Sumedang, Jawa Barat, Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) langsung mengambil langkah terukur. Utamanya, melakukan survei dan kajian mendalam di lokasi yang telah ditentukan untuk pengambilan data.\nPlt. Deputi Bidang Geofisika Hanif Andi Nugraha menyebutkan fokus utama lokasi berada di Kecamatan Cimalaka, Sumedang Utara dan Selatan. Lokasi ini dipilih karena mengalami dampak signifikan terjadinya aktivitas gempabumi beberapa waktu lalu. Adapun tim yang turun meliputi Pusat Seismologi Teknik, Stasiun Geofisika Bandung, Stasiun Geofisika Tangerang, dan Balai Besar MKG Wilayah II.\nTim BMKG memulai survei dengan mendeteksi dan memahami perkembangan aktivitas gempa susulan yang terjadi. Seismisitas menjadi pusat perhatian utama karena memungkinkan identifikasi jalur sesar dan mekanisme sumber gempa. Melalui survei makroseismik, BMKG memetakan sebaran kondisi dampak kerusakan dan memverifikasi tingkat guncangan tanah di pelbagai lokasi terdampak.\nDalam rangka mendukung perencanaan wilayah yang lebih aman, BMKG melakukan survei mikrozonasi. Pemetaan sebaran dan intensitas tingkat guncangan tanah setempat menjadi landasan bagi penyempurnaan Rencana Tata Ruang Wilayah dan aturan bangunan tahan gempa.\nIdentifikasi perubahan permukaan tanah akibat gempabumi menjadi fokus selanjutnya dengan survei deformasi permukaan menjadi landasan penting. Langkah ini membantu dalam mengidentifikasi potensi risiko gempabumi di masa depan, memberikan pemahaman yang lebih luas terkait jalur sesar.\nSelain itu, teknologi Drone Lidar menjadi salah satu upaya BMKG dalam pemetaan sebaran tingkat kerusakan dan kondisi morfotektonik. Dalam rancangan area terbangnya, drone Lidar melayang di atas area seluas 3.250 hektare selama lima hari untuk mengumpulkan data fotogrametri dan Digital Elevation Model (DEM).\nLangkah terakhir melibatkan evaluasi morfotektonik dari hasil survei makroseismik, seismisitas, dan deformasi permukaan digabungkan. Proses ini memantapkan identifikasi dan validasi jalur sesar, memberikan pemahaman yang lebih holistik mengenai kejadian gempabumi tersebut.\nMelalui langkah-langkah ini, BMKG tidak hanya memberikan pemahaman yang mendalam tentang dampak gempabumi, tetapi juga memberikan dasar yang kuat untuk upaya mitigasi bencana di masa depan. Kesigapan dan komitmen BMKG memberikan harapan bagi keberlanjutan dan keselamatan masyarakat Sumedang.',
            'image_url': '/static/images/gambar4.jpg',
            'date_posted': datetime(2023, 1, 7)
        }
    ]
    for article in articles:
        new_article = Article(
            title=article['title'],
            content=article['content'],
            image_url=article['image_url'],
            date_posted=article['date_posted']
        )
        db.session.add(new_article)
    db.session.commit()

# Variabel untuk mengecek apakah inisialisasi sudah dilakukan
initialized = False

@app.before_request
def initialize_database():
    global initialized
    if not initialized:
        db.create_all()
        if Article.query.count() == 0:
            add_default_articles()
        initialized = True

# Endpoint untuk melihat artikel berdasarkan ID
@app.route('/article/<int:article_id>')
def view_article(article_id):
    article = Article.query.get_or_404(article_id)
    return render_template('view_article.html', article=article)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_message = request.json.get('message')
        conn = http.client.HTTPSConnection(RAPIDAPI_HOST)
        payload = json.dumps({
            "messages": [{"role": "user", "content": user_message}],
            "system_prompt": "",
            "temperature": 0.9,
            "top_k": 5,
            "top_p": 0.9,
            "image": "",
            "max_tokens": 256
        })
        headers = {
            'x-rapidapi-key': RAPIDAPI_KEY,
            'x-rapidapi-host': RAPIDAPI_HOST,
            'Content-Type': 'application/json'
        }
        conn.request("POST", RAPIDAPI_URL, payload, headers)
        res = conn.getresponse()
        data = res.read()
        response_data = json.loads(data.decode("utf-8"))

        if res.status == 200:
            print(response_data)  # Tambahkan ini untuk melihat respons API
            if 'result' in response_data:
                print(f"User: {user_message}")
                print(f"Bot: {response_data['result']}")
                return jsonify({'response': response_data['result']})
            else:
                print('Invalid response format:', response_data)
                return jsonify({'error': 'Invalid response format from API'}), 500
        else:
            print(f'Error: {res.status} - {response_data}')
            return jsonify({'error': 'Error fetching response from API'}), 500
    else:
        return render_template('chat.html')


if __name__ == '__main__':
    app.run(debug=True)
