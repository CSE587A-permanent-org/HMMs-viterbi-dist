# You can change these variables to use a different base image, but
# you must ensure that your base image inherits from one of ours.
# You can also override these at build time with --build-arg flags
ARG BASE_REPO=gradescope/autograder-base
ARG TAG=latest
ARG PLATFORM=linux/amd64 
FROM --platform=${PLATFORM} ${BASE_REPO}:${TAG} AS builder

# Install system build tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl  \
    tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Poetry system-wide
ENV POETRY_HOME=/usr/local
RUN curl -sSL https://install.python-poetry.org | python3
RUN echo poetry --version
RUN poetry config virtualenvs.create false

# Copy dependency files first for caching
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry lock
RUN poetry install --no-root --no-interaction --no-ansi

FROM --platform=${PLATFORM} ${BASE_REPO}:${TAG} AS runtime

# Copy Poetry binary from builder stage
COPY --from=builder /usr/local/bin/poetry /usr/local/bin/poetry

# (Optional) Copy Poetry cache/config if you want to run `poetry` in runtime
COPY --from=builder /root/.cache/pypoetry /root/.cache/pypoetry
COPY --from=builder /root/.config/pypoetry /root/.config/pypoetry

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local /usr/local

# This is the stuff from gradescope for the autograder
ADD autograder/source /autograder/source

# This is where the results of the tests are stored
RUN mkdir /autograder/results

ADD assignment/test_assignment.py /autograder/source/

RUN cp /autograder/source/run_autograder /autograder/run_autograder

# Ensure that scripts are Unix-friendly and executable
# RUN dos2unix /autograder/run_autograder
RUN chmod +x /autograder/run_autograder

# Set the timezone to UTC
ENV TZ=UTC

# Set the system time in the container
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# symlink python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python