# Use the official Rust image as the base image
FROM rust:latest as builder

# Set the working directory
WORKDIR /usr/src/app

# Copy the Cargo.toml and Cargo.lock files into the container
COPY Cargo.toml Cargo.lock ./

# Create a dummy source file to build the dependencies
RUN mkdir -p src && echo "fn main() {}" > src/main.rs && cargo build --release

# Copy the actual source code into the container
COPY src ./src

# Build the project in release mode
RUN cargo build --release

# Run the compiled binary in a minimal environment
FROM debian:buster-slim

# Install any necessary dependencies for the binary
RUN apt-get update && apt-get install -y libssl-dev ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy the binary from the builder stage
COPY --from=builder /usr/src/app/target/release/mcts_chess /usr/local/bin/mcts_chess

# Set the entrypoint to your binary
ENTRYPOINT ["mcts_chess"]