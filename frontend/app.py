import streamlit as st
import requests
import pandas as pd
import io
import PyPDF2
import json

st.set_page_config(page_title="LangGraph ChatBot")
st.title("LangGraph Copilot")

uploaded_file = st.file_uploader("Nahrajte soubor (XLSX, CSV, PDF)", type=["xlsx", "csv", "pdf"])

if uploaded_file is not None:
    file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type}
    st.write("Nahraný soubor:", file_details["filename"])
    
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        df = pd.DataFrame({"text": [text]})
    
    st.write("Náhled dat:")
    st.dataframe(df.head())
    
    st.subheader("LEAN analýza")
    
    if all(col in df.columns for col in ['availability', 'performance', 'quality']):
        oee = df['availability'] * df['performance'] * df['quality']
        st.write("**Celková efektivnost zařízení (OEE):**")
        st.write(f"- Průměrná hodnota: {oee.mean():.2%}")
        st.write(f"- Trend: {'rostoucí' if oee.iloc[-1] > oee.iloc[0] else 'klesající'}")
        st.line_chart(oee)
    
    if 'downtime' in df.columns and 'downtime_reason' in df.columns:
        st.write("**Paretova analýza odstávek:**")
        downtime_analysis = df.groupby('downtime_reason')['downtime'].agg(['sum', 'count']).sort_values('sum', ascending=False)
        st.dataframe(downtime_analysis)
        st.bar_chart(downtime_analysis['sum'])
    
    if 'defects' in df.columns and 'total_production' in df.columns:
        defect_rate = df['defects'] / df['total_production']
        st.write("**Míra defektů:**")
        st.write(f"- Průměrná hodnota: {defect_rate.mean():.2%}")
        st.write(f"- Trend: {'rostoucí' if defect_rate.iloc[-1] > defect_rate.iloc[0] else 'klesající'}")
        st.line_chart(defect_rate)
    
    if 'maintenance_time' in df.columns and 'production_time' in df.columns:
        maintenance_ratio = df['maintenance_time'] / df['production_time']
        st.write("**Poměr údržby k produkčnímu času:**")
        st.write(f"- Průměrná hodnota: {maintenance_ratio.mean():.2%}")
        st.write(f"- Trend: {'rostoucí' if maintenance_ratio.iloc[-1] > maintenance_ratio.iloc[0] else 'klesající'}")
        st.line_chart(maintenance_ratio)
    
    if 'waste' in df.columns and 'total_material' in df.columns:
        waste_rate = df['waste'] / df['total_material']
        st.write("**Míra odpadu:**")
        st.write(f"- Průměrná hodnota: {waste_rate.mean():.2%}")
        st.write(f"- Trend: {'rostoucí' if waste_rate.iloc[-1] > waste_rate.iloc[0] else 'klesající'}")
        st.line_chart(waste_rate)
    
    st.session_state["uploaded_data"] = df.to_json()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Zadejte zprávu...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Přemýšlím..."):
            try:
                request_data = {"message": user_input}
                if "uploaded_data" in st.session_state:
                    request_data["data"] = st.session_state["uploaded_data"]
                
                response = requests.post("http://localhost:8000/chat", json=request_data)
                
                if response.status_code != 200:
                    st.error(f"Chyba serveru: {response.status_code}")
                    output = "Omlouvám se, došlo k chybě na serveru."
                else:
                    try:
                        response_data = response.json()
                        output = response_data["response"]
                        
                        if "suggestions" in response_data and response_data["suggestions"]:
                            st.markdown("**Návrhy na další LEAN analýzu:**")
                            for suggestion in response_data["suggestions"]:
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(suggestion["question"])
                                with col2:
                                    if st.button("Ano", key=f"yes_{suggestion['type']}_{suggestion.get('reason', '')}"):
                                        if suggestion["type"] == "oee_change":
                                            follow_up = f"Prosím, analyzuj příčiny změny v OEE ({suggestion['change']:.1f}%). Co způsobilo tuto změnu?"
                                        elif suggestion["type"] == "downtime_reason":
                                            follow_up = f"Prosím, analyzuj příčinu odstávek '{suggestion['reason']}' ({suggestion['duration']:.1f} hodin). Jaké jsou možnosti optimalizace?"
                                        else:
                                            follow_up = f"Prosím, analyzuj příčiny změny v míře defektů ({suggestion['change']:.1f}%). Co způsobilo tuto změnu?"
                                        
                                        st.session_state.messages.append({"role": "user", "content": follow_up})
                                        st.experimental_rerun()
                                    
                                    if st.button("Ne", key=f"no_{suggestion['type']}_{suggestion.get('reason', '')}"):
                                        st.markdown("Děkuji za odpověď.")
                    except requests.exceptions.JSONDecodeError:
                        st.error("Nepodařilo se zpracovat odpověď ze serveru")
                        print("RAW RESPONSE:", response.text)
                        output = "Omlouvám se, došlo k chybě při zpracování odpovědi."
            except requests.exceptions.ConnectionError:
                st.error("Nelze se připojit k serveru. Zkontrolujte, zda je backend spuštěn.")
                output = "Omlouvám se, server není dostupný."
            
            st.markdown(output)

    st.session_state.messages.append({"role": "assistant", "content": output})