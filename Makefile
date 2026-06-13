build:
	cargo build --release
	mkdir -p dist
	cp target/release/openai-proxy dist/openai-proxy

linux:
	cargo build --release --target x86_64-unknown-linux-gnu
	mkdir -p dist/linux-x86
	cp target/x86_64-unknown-linux-gnu/release/openai-proxy dist/linux-x86/openai-proxy

clean:
	cargo clean
	rm -rf dist

.PHONY: build linux clean
