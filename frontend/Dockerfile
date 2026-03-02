FROM nginx:alpine

WORKDIR /app

COPY frontend/. .

ARG NGINX_CONFIG
COPY ${NGINX_CONFIG} /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]