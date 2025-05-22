# app.py  –  Modelo Predictivo para la Prevención de Adicciones
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------- 1. Cargar pipeline ------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_logreg_tuned.pkl")
model = load_model()



# ---------- 2. Diccionarios de mapeo -----------------------------------------
likert_score = {'Nunca':0,'Rara vez':1,'A veces':2,'Con frecuencia':3,'Siempre':4}
likert_perc  = {'Nunca':0,'Rara vez':1,'A veces':2,'La mayoría del tiempo':3,'Siempre':4}

mapa_genero  = {'Masculino':0,'Femenino':1,'No binario':2}
mapa_carreras = {
    'Ingeniería en Inteligencia de Datos y Ciberseguridad':0,'Ingeniería Mecánica':1,
    'Ingeniería Mecatrónica':2,'Ingeniería Industrial e Innovación Basada en Datos':3,
    'Ingeniería en Animación y Videojuegos':4,'Ingeniería en Innovación y Diseño':5,
    'Matematicás Aplicadas':6,'Administración y Finanzas':7,'Administración y Negocios Internacionales':8,
    'Administración y Dirección':9,'Administración y Mercadotecnia':10,'Business Intelligence':11,
    'Contaduría':12,'Administración y Hospitalidad ESDAI':13,'Dirección de Negocios Gastronómicos':14,
    'Medicina':15,'Comunicación':16,'Música e Innovación':17,'Filosofía':18,'Derecho':19,'Psicología':20,'Pedagogía':21
}
mapa_trabajo = {'No':0,'Sí, medio tiempo':1,'Sí, tiempo completo':1}
mapa_beca   = {'No':0,'Sí, académica':1,'Sí, económica':2,'Sí, cultural/deportiva':3,'Sí, de trabajo':4}
mapa_dsm01  = {'No consumo':0,'No, se ha mantenido igual o disminuido':0,
               'Ha aumentado ligeramente':1,'Ha aumentado moderadamente':3,'Ha aumentado mucho':4}
mapa_dsm08  = {'No':0,'Sí':4}
mapa_bin    = {'No':0,'Sí':1}
mapa_eventos= {'Ninguno':0,'Uno':1,'Dos a tres':2,'Cuatro o más':3}
mapa_padres = {'Con ambos padres':0,'Solo con madre':1,'Solo con padre':2,
               'Con ningún padre/madre':3,'Otro (abuelos, tutores, etc.)':4}
mapa_rend   = {'Malo':0,'Regular':1,'Bueno':2,'Excelente':3}
mapa_estres = {'Nada de estrés':0,'Poco estrés':1,'Estrés moderado':2,'Mucho estrés':3,'Estrés extremo':4}
mapa_salud  = {'Malo':0,'Regular':1,'Bueno':2,'Excelente':3}

# ---------- 3. Orden exacto de columnas --------------------------------------
col_order = [
    'age','gender','career','works','scholarship_type','age_first_alcohol','age_first_drug',
    'dsm_01_consume_rise','dsm_02_reduce_attempt','dsm_03_life_consume','dsm_04_craving',
    'dsm_05_responsabilities','dsm_06_relationship_strain','dsm_07_act_reduct','dsm_08_urgency',
    'dsm_09_consume_harm','dsm_10_tolerance','dsm_11_abstinence','consume_trouble','worry_control',
    'family_addiction_issues','bullying','childhood_trauma','parent_relationship','safe_home_chldhd',
    'childhood_physical_punishments','academic_performance','academic_stress','physical_health',
    'frequently_anxious','isolation','incomprehended','friendship_trouble','random_sadness'
]

# ---------- 4. Interfaz Streamlit -------------------------------------------
st.title("Modelo Predictivo para la Prevención de Adicciones")

with st.form("predict_form"):
    st.subheader("Completa el cuestionario")

    age = st.number_input("Edad", min_value=15, max_value=40, value=18, step=1)

    gender = st.selectbox("Género", list(mapa_genero.keys()))
    career = st.selectbox("Carrera", list(mapa_carreras.keys()))
    works  = st.selectbox("¿Trabajas actualmente mientras estudias?", list(mapa_trabajo.keys()))
    scholarship = st.selectbox("Tipo de beca universitaria", list(mapa_beca.keys()))

    age_alcohol = st.number_input("Edad primer consumo de alcohol (0 si nunca)", min_value=0, max_value=40, value=0, step=1)
    age_drug    = st.number_input("Edad primer consumo de droga ilícita (0 si nunca)", min_value=0, max_value=40, value=0, step=1)

    # --- DSM-5 items (solo ejemplos, repite según sea necesario) -------------
    dsm01 = st.selectbox("¿Tu consumo ha aumentado con el tiempo?", list(mapa_dsm01.keys()))
    dsm02 = st.selectbox("Intentos fallidos por reducir consumo", list(likert_score.keys()))
    dsm03 = st.selectbox("¿Tu vida gira alrededor del consumo?", list(likert_score.keys()))
    dsm04 = st.selectbox("Frecuencia de craving en 3 meses", list(likert_score.keys()))
    dsm05 = st.selectbox("Responsabilidades afectadas por consumo", list(likert_score.keys()))
    dsm06 = st.selectbox("Relaciones deterioradas por consumo", list(likert_score.keys()))
    dsm07 = st.selectbox("Actividades reducidas por consumo", list(likert_score.keys()))
    dsm08 = st.selectbox("¿Atención médica de urgencia por intoxicación?", list(mapa_dsm08.keys()))
    dsm09 = st.selectbox("Continúa consumiendo pese a daño físico", list(likert_score.keys()))
    dsm10 = st.selectbox("Necesita más sustancia para mismo efecto", list(likert_score.keys()))
    dsm11 = st.selectbox("Síntomas de abstinencia", list(likert_score.keys()))

    consume_trouble = st.selectbox("¿Ha tenido problemas por consumo?", list(mapa_bin.keys()))
    worry_control   = st.selectbox("¿Preocupado por perder control?", list(likert_score.keys()))
    fam_addict      = st.selectbox("Familiar con problemas de adicción", list(mapa_bin.keys()))
    bullying        = st.selectbox("¿Sufrió bullying en infancia/adolescencia?", list(mapa_bin.keys()))
    trauma          = st.selectbox("Eventos dolorosos en infancia", list(mapa_eventos.keys()))
    parent_rel      = st.selectbox("Convivencia con padres en infancia", list(mapa_padres.keys()))
    safe_home       = st.selectbox("¿Hogar seguro en infancia?", list(likert_perc.keys()))
    punish          = st.selectbox("Castigos físicos en infancia", list(likert_perc.keys()))
    academic_perf   = st.selectbox("Rendimiento académico reciente", list(mapa_rend.keys()))
    academic_stress = st.selectbox("Nivel de estrés académico", list(mapa_estres.keys()))
    health          = st.selectbox("Salud física actual", list(mapa_salud.keys()))
    anxious         = st.selectbox("¿Ansioso frecuentemente?", list(mapa_bin.keys()))
    isolation       = st.selectbox("¿Se aísla cuando tiene problemas?", list(likert_score.keys()))
    incomprehended  = st.selectbox("¿Siente que nadie lo comprende?", list(likert_score.keys()))
    friendship      = st.selectbox("Dificultad para mantener amistades", list(likert_score.keys()))
    sadness         = st.selectbox("Tristeza sin razón aparente", list(likert_score.keys()))

    submitted = st.form_submit_button("Calcular riesgo")

# ---------- 5. Procesar y predecir ------------------------------------------
if submitted:
    # Construir registro único
    row = {
        'age': age,
        'gender': mapa_genero[gender],
        'career': mapa_carreras[career],
        'works': mapa_trabajo[works],
        'scholarship_type': mapa_beca[scholarship],
        'age_first_alcohol': np.nan if age_alcohol==0 else age_alcohol,
        'age_first_drug': np.nan if age_drug==0 else age_drug,
        'dsm_01_consume_rise': mapa_dsm01[dsm01],
        'dsm_02_reduce_attempt': likert_score[dsm02],
        'dsm_03_life_consume': likert_score[dsm03],
        'dsm_04_craving': likert_score[dsm04],
        'dsm_05_responsabilities': likert_score[dsm05],
        'dsm_06_relationship_strain': likert_score[dsm06],
        'dsm_07_act_reduct': likert_score[dsm07],
        'dsm_08_urgency': mapa_dsm08[dsm08],
        'dsm_09_consume_harm': likert_score[dsm09],
        'dsm_10_tolerance': likert_score[dsm10],
        'dsm_11_abstinence': likert_score[dsm11],
        'consume_trouble': mapa_bin[consume_trouble],
        'worry_control': likert_score[worry_control],
        'family_addiction_issues': mapa_bin[fam_addict],
        'bullying': mapa_bin[bullying],
        'childhood_trauma': mapa_eventos[trauma],
        'parent_relationship': mapa_padres[parent_rel],
        'safe_home_chldhd': likert_perc[safe_home],
        'childhood_physical_punishments': likert_perc[punish],
        'academic_performance': mapa_rend[academic_perf],
        'academic_stress': mapa_estres[academic_stress],
        'physical_health': mapa_salud[health],
        'frequently_anxious': mapa_bin[anxious],
        'isolation': likert_score[isolation],
        'incomprehended': likert_score[incomprehended],
        'friendship_trouble': likert_score[friendship],
        'random_sadness': likert_score[sadness],
    }

    # Reordenar y convertir a DataFrame
    X_new = pd.DataFrame([row])[col_order]

    # Predecir clase y proba
    pred_class = int(model.predict(X_new)[0])
    pred_proba = model.predict_proba(X_new)[0]

    risk_labels = ["No riesgo", "Riesgo leve", "Riesgo moderado", "Riesgo grave"]
    st.markdown(f"### Resultado: **{risk_labels[pred_class]}**")
    st.write("Probabilidades:", 
             {risk_labels[i]: f"{p:.2%}" for i, p in enumerate(pred_proba)})

    st.success("Evaluación completada • Gracias por participar")
