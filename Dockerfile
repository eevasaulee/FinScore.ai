# 1. База: официальный Python
FROM python:3.11-slim

# 2. Рабочая директория внутри контейнера
WORKDIR /app

# 3. Копируем файлы зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 4. Копируем весь код приложения
COPY . .

# 5. Указываем порт, на котором будет работать Uvicorn
EXPOSE 8000

# 6. По умолчанию запускаем API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
