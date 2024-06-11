FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# do stuff here
WORKDIR /app

#COPY [ "model2.bin", "model.bin" ]

COPY ["starter.py","requisitos.txt","./"]

RUN pip install --no-cache-dir -r requisitos.txt

CMD ["python","starter.py","2022","4"]