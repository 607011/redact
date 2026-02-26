FROM nginx:alpine

WORKDIR /app

COPY frontend/index.html .
COPY frontend/50x.html .
COPY frontend/404.html .
RUN mkdir -p static/fonts
COPY frontend/static/fonts/*.woff2 ./static/fonts

ARG NGINX_CONFIG
COPY ${NGINX_CONFIG} /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]