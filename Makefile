BINARY=ollama-bridge
VERSION=$(shell git describe --tags --always 2>/dev/null || echo "dev")

.PHONY: all build clean test

all: build

build:
	@echo "Building $(BINARY) $(VERSION)..."
	go build -ldflags="-X main.Version=$(VERSION)" -o $(BINARY) .

clean:
	rm -f $(BINARY)

test:
	go test ./...
