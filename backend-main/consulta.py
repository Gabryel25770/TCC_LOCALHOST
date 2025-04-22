from db_models import SessionLocal, Registro

def listar_registros():
    db = SessionLocal()
    resultados = db.query(Registro).all()

    for item in resultados:
        print(f"ID: {item.id} | Texto: {item.texto} | Sentimento: {item.sentimento}")

    db.close()

if __name__ == "__main__":
    listar_registros()
