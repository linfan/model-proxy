build:
	cargo build --release
	mkdir -p dist
	cp target/release/model-proxy dist/model-proxy

linux:
	cargo build --release --target x86_64-unknown-linux-gnu
	mkdir -p dist/linux-x86
	cp target/x86_64-unknown-linux-gnu/release/model-proxy dist/linux-x86/model-proxy

clean:
	cargo clean
	rm -rf dist

.PHONY: build linux clean
