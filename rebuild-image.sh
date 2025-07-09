#!/bin/bash


docker kill tilus-artifacts
docker rm tilus-artifacts
docker build -t tilus-artifacts:latest .
