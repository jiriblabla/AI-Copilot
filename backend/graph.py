from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import pandas as pd
import json
from io import StringIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


try:
    llm = ChatOpenAI()
    embeddings = OpenAIEmbeddings()
except Exception as e:
    logger.error(f"Chyba při inicializaci modelů: {str(e)}")
    raise


class GraphState(BaseModel):
    messages: list
    data: str | None = None
    vectorstore: FAISS = None
    analysis: dict | None = None
    metrics: dict | None = None
    lean_metrics: dict | None = None

    class Config:
        arbitrary_types_allowed = True


def analyze_metrics(df):
    try:
        metrics = {}
        
        employee_columns = [col for col in df.columns if 'employee' in col.lower() or 'worker' in col.lower() or 'operator' in col.lower()]
        if employee_columns:
            metrics['employee_performance'] = analyze_employee_performance(df, employee_columns)
        
        downtime_columns = [col for col in df.columns if 'downtime' in col.lower() or 'status' in col.lower() or 'stop' in col.lower()]
        if downtime_columns:
            metrics['downtime_analysis'] = analyze_downtime(df, downtime_columns)
        
        quality_columns = [col for col in df.columns if 'quality' in col.lower() or 'error' in col.lower() or 'defect' in col.lower()]
        if quality_columns:
            metrics['quality_analysis'] = analyze_quality(df, quality_columns)
        
        productivity_columns = [col for col in df.columns if 'output' in col.lower() or 'productivity' in col.lower() or 'time' in col.lower()]
        if productivity_columns:
            metrics['productivity_analysis'] = analyze_productivity(df, productivity_columns)
        
        return metrics
    except Exception as e:
        logger.error(f"Chyba při analýze metrik: {str(e)}")
        return None

def analyze_employee_performance(df, employee_columns):
    try:
        analysis = {}
        for col in employee_columns:
            employee_stats = df.groupby(col).agg({
                'output': ['mean', 'sum', 'count'],
                'quality': ['mean', 'sum'],
                'downtime': ['sum', 'count']
            }).round(2)
            
            top_performers = employee_stats.nlargest(5, ('output', 'mean'))
            bottom_performers = employee_stats.nsmallest(5, ('output', 'mean'))
            
            analysis[col] = {
                'summary': f"Analýza výkonnosti pro {col}",
                'top_performers': top_performers.to_dict(),
                'bottom_performers': bottom_performers.to_dict(),
                'overall_stats': employee_stats.describe().to_dict()
            }
        return analysis
    except Exception as e:
        logger.error(f"Chyba při analýze výkonnosti zaměstnanců: {str(e)}")
        return None

def analyze_downtime(df, downtime_columns):
    try:
        analysis = {}
        for col in downtime_columns:
            downtime_stats = df[df[col] > 0].groupby(pd.Grouper(key='timestamp', freq='D'))[col].agg(['sum', 'count', 'mean'])
            
            if 'downtime_reason' in df.columns:
                reason_stats = df.groupby('downtime_reason')[col].agg(['sum', 'count']).sort_values('sum', ascending=False)
                analysis['reasons'] = reason_stats.head(5).to_dict()
            
            analysis[col] = {
                'summary': f"Analýza odstávek pro {col}",
                'daily_stats': downtime_stats.to_dict(),
                'total_downtime': df[col].sum(),
                'avg_downtime': df[col].mean()
            }
        return analysis
    except Exception as e:
        logger.error(f"Chyba při analýze odstávek: {str(e)}")
        return None

def analyze_quality(df, quality_columns):
    try:
        analysis = {}
        for col in quality_columns:
            quality_stats = df.groupby(pd.Grouper(key='timestamp', freq='D'))[col].agg(['mean', 'sum', 'count'])
            
            quality_trend = df[col].rolling(window=7).mean()
            
            analysis[col] = {
                'summary': f"Analýza kvality pro {col}",
                'daily_stats': quality_stats.to_dict(),
                'quality_trend': quality_trend.to_dict(),
                'defect_rate': (df[col] > 0).mean()
            }
        return analysis
    except Exception as e:
        logger.error(f"Chyba při analýze kvality: {str(e)}")
        return None

def analyze_productivity(df, productivity_columns):
    try:
        analysis = {}
        for col in productivity_columns:
            productivity_stats = df.groupby(pd.Grouper(key='timestamp', freq='D'))[col].agg(['mean', 'sum', 'count'])
            
            productivity_trend = df[col].rolling(window=7).mean()
            
            analysis[col] = {
                'summary': f"Analýza produktivity pro {col}",
                'daily_stats': productivity_stats.to_dict(),
                'productivity_trend': productivity_trend.to_dict(),
                'efficiency': df[col].mean() / df[col].max() if df[col].max() > 0 else 0
            }
        return analysis
    except Exception as e:
        logger.error(f"Chyba při analýze produktivity: {str(e)}")
        return None

def analyze_lean_metrics(df):
    try:
        lean_metrics = {}
        
        if all(col in df.columns for col in ['availability', 'performance', 'quality']):
            lean_metrics['oee'] = {
                'value': df['availability'] * df['performance'] * df['quality'],
                'trend': (df['availability'] * df['performance'] * df['quality']).rolling(window=7).mean(),
                'summary': "Analýza celkové efektivnosti zařízení (OEE)"
            }
        
        if 'downtime' in df.columns and 'downtime_reason' in df.columns:
            downtime_analysis = df.groupby('downtime_reason')['downtime'].agg(['sum', 'count']).sort_values('sum', ascending=False)
            lean_metrics['downtime_pareto'] = {
                'data': downtime_analysis.to_dict(),
                'summary': "Paretova analýza příčin odstávek"
            }
        
        if 'defects' in df.columns and 'total_production' in df.columns:
            lean_metrics['defect_rate'] = {
                'value': df['defects'] / df['total_production'],
                'trend': (df['defects'] / df['total_production']).rolling(window=7).mean(),
                'summary': "Analýza míry defektů"
            }
        
        if 'maintenance_time' in df.columns and 'production_time' in df.columns:
            lean_metrics['maintenance_ratio'] = {
                'value': df['maintenance_time'] / df['production_time'],
                'trend': (df['maintenance_time'] / df['production_time']).rolling(window=7).mean(),
                'summary': "Analýza poměru údržby k produkčnímu času"
            }
        
        if 'waste' in df.columns and 'total_material' in df.columns:
            lean_metrics['waste_rate'] = {
                'value': df['waste'] / df['total_material'],
                'trend': (df['waste'] / df['total_material']).rolling(window=7).mean(),
                'summary': "Analýza míry odpadu"
            }
        
        return lean_metrics
    except Exception as e:
        logger.error(f"Chyba při LEAN analýze: {str(e)}")
        return None

def process_data(data_str: str):
    try:
        df = pd.read_json(StringIO(data_str))
        logger.info(f"Data úspěšně načtena, velikost: {df.shape}")
        
        analysis = analyze_extremes(df)
        
        lean_metrics = analyze_lean_metrics(df)
        
        if isinstance(df, pd.DataFrame):
            df_sample = df.head(1000)
            text = df_sample.to_string()
        else:
            text = str(df)
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Text rozdělen na {len(chunks)} částí")
        
        vectorstore = FAISS.from_texts(chunks, embeddings)
        logger.info("Vektorová databáze úspěšně vytvořena")
        return vectorstore, analysis, lean_metrics
    except Exception as e:
        logger.error(f"Chyba při zpracování dat: {str(e)}")
        return None, None, None


def analyze_extremes(df):
    try:
        analysis = {}
        for column in df.select_dtypes(include=['number']).columns:
            mean = df[column].mean()
            std = df[column].std()
            
            extremes = df[abs(df[column] - mean) > 3 * std]
            
            if not extremes.empty:
                top_extremes = extremes.nlargest(5, column) if mean > 0 else extremes.nsmallest(5, column)
                
                analysis[column] = {
                    'mean': mean,
                    'std': std,
                    'extremes': top_extremes.to_dict('records'),
                    'count': len(extremes),
                    'summary': f"Sloupec {column} má {len(extremes)} extrémních hodnot. Průměr: {mean:.2f}, Směrodatná odchylka: {std:.2f}"
                }
        
        return analysis
    except Exception as e:
        logger.error(f"Chyba při analýze extrémních hodnot: {str(e)}")
        return None

def generate_lean_suggestions(df, lean_metrics):
    suggestions = []
    
    if 'oee' in lean_metrics:
        oee_trend = lean_metrics['oee']['trend']
        if len(oee_trend) > 1:
            last_change = ((oee_trend.iloc[-1] - oee_trend.iloc[-2]) / oee_trend.iloc[-2]) * 100
            if abs(last_change) > 10:
                suggestions.append({
                    'type': 'oee_change',
                    'change': last_change,
                    'question': f"Detekoval jsem významnou změnu v OEE ({last_change:.1f}%). Chcete analyzovat příčiny této změny?"
                })
    
    if 'downtime_pareto' in lean_metrics:
        top_reasons = list(lean_metrics['downtime_pareto']['data'].items())[:3]
        for reason, data in top_reasons:
            suggestions.append({
                'type': 'downtime_reason',
                'reason': reason,
                'duration': data['sum'],
                'question': f"Nejčastější příčinou odstávek je '{reason}' ({data['sum']:.1f} hodin). Chcete analyzovat tuto příčinu podrobněji?"
            })

    if 'defect_rate' in lean_metrics:
        defect_trend = lean_metrics['defect_rate']['trend']
        if len(defect_trend) > 1:
            last_change = ((defect_trend.iloc[-1] - defect_trend.iloc[-2]) / defect_trend.iloc[-2]) * 100
            if abs(last_change) > 5:
                suggestions.append({
                    'type': 'defect_rate_change',
                    'change': last_change,
                    'question': f"Detekoval jsem významnou změnu v míře defektů ({last_change:.1f}%). Chcete analyzovat příčiny této změny?"
                })
    
    return suggestions

def respond(state: GraphState):
    try:
        history = []
    
        if state.data and not state.vectorstore:
            logger.info("Zpracovávám data a vytvářím vektorovou databázi")
            state.vectorstore, state.analysis, state.lean_metrics = process_data(state.data)
            
            if state.analysis and state.lean_metrics:
                state.suggestions = generate_lean_suggestions(
                    pd.read_json(StringIO(state.data)),
                    state.lean_metrics
                )
        
        system_message = """Jsi asistent, agent, který analyzuje firmní data pomocí LEAN metodologie.
        Tvým úkolem je:
        1. Identifikovat a analyzovat klíčové LEAN metriky:
           - OEE (Overall Equipment Effectiveness)
           - Paretova analýza odstávek
           - Míra defektů
           - Poměr údržby k produkčnímu času
           - Míra odpadu
        2. Detekovat významné změny v metrikách (>10% pro OEE, >5% pro ostatní)
        3. Analyzovat příčiny odstávek a navrhnout optimalizace
        4. Identifikovat druhy plýtvání (Defects, Excess Inventory, Overproduction, Waiting, Excess Movement, Overprocessing, Transportation)
        5. Poskytovat doporučení pro kontinuální zlepšování
        6. Generovat interaktivní návrhy na další analýzu
        
        Při analýze se zaměř na:
        - Kontext celkové výroby
        - Příčiny odstávek a jejich dopad
        - Kvalitu výroby a míru defektů
        - Efektivitu údržby strojů
        - Míru odpadu a plýtvání
        - Trendy v klíčových metrikách
        
        Pro každý významný nález nabídni možnost podrobnější analýzy."""
        
        if state.analysis:
            analysis_summary = "\n".join([v['summary'] for v in state.analysis.values()])
            system_message += f"\n\nAnalýza extrémních hodnot:\n{analysis_summary}"
            
        if state.lean_metrics:
            lean_summary = "\n".join([
                f"\n{metric_type}:\n{v['summary']}"
                for metric_type, v in state.lean_metrics.items()
            ])
            system_message += f"\n\nLEAN analýza:{lean_summary}"
            
        history.append(SystemMessage(content=system_message))
        
        if state.vectorstore and state.messages:
            last_message = state.messages[-1]["content"]
            logger.info(f"Hledám relevantní kontext pro otázku: {last_message}")
            docs = state.vectorstore.similarity_search(last_message, k=2)
            context = "\n".join([doc.page_content for doc in docs])
            history.append(SystemMessage(content=f"Relevantní kontext z dat:\n{context}"))
        
        recent_messages = state.messages[-5:] if len(state.messages) > 5 else state.messages
        
        for msg in recent_messages:
            if msg["type"] == "human":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                history.append(AIMessage(content=msg["content"]))
        
        logger.info("Generuji odpověď")
        response = llm.invoke(history)
        
        response_content = response.content
        if hasattr(state, 'suggestions') and state.suggestions:
            response_content += "\n\n**Návrhy na další LEAN analýzu:**\n"
            for suggestion in state.suggestions:
                response_content += f"- {suggestion['question']}\n"
        
        return {"messages": state.messages + [{"type": "ai", "content": response_content}], 
                "data": state.data,
                "vectorstore": state.vectorstore,
                "analysis": state.analysis,
                "lean_metrics": state.lean_metrics,
                "suggestions": state.suggestions if hasattr(state, 'suggestions') else None}
    except Exception as e:
        logger.error(f"Chyba při generování odpovědi: {str(e)}")
        return {"messages": state.messages + [{"type": "ai", "content": "Omlouvám se, došlo k chybě při generování odpovědi."}], 
                "data": state.data,
                "vectorstore": state.vectorstore,
                "analysis": state.analysis,
                "lean_metrics": state.lean_metrics,
                "suggestions": None}


def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("chat", RunnableLambda(respond))
    builder.set_entry_point("chat")
    builder.set_finish_point("chat")

    return builder.compile()