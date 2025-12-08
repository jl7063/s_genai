# Dockerfile
FROM python:3.11-slim

# 安装基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目全部代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动 FastAPI 服务器（就是你这次作业要交的 API）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
