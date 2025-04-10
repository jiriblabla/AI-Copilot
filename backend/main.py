from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from backend.graph import build_graph, GraphState
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        message = data["message"]
        uploaded_data = data.get("data")

        state = {
            "messages": [{"type": "human", "content": message}],
            "data": uploaded_data
        }
        result = await graph.ainvoke(state)


        last_msg = result["messages"][-1]["content"]
        return {"response": last_msg}
    except Exception as e:
        print(f"Chyba v endpointu: {str(e)}")
        return {"response": "Omlouvám se, došlo k chybě při zpracování požadavku."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)