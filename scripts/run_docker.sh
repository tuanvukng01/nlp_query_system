#!/usr/bin/env bash
docker build -t nlp-query-system:latest ../api
docker run -d -p 8080:5000 nlp-query-system:latest