FROM nginx:alpine

WORKDIR /app

COPY frontend/index.html .
RUN mkdir -p static/fonts
COPY frontend/static/fonts ./static/fonts

ARG NGINX_CONFIG_PATH
COPY ${NGINX_CONFIG_PATH}/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]