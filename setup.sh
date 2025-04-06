#!/bin/bash
mkdir -p ~/.streamlit/

echo "\
[general]
email = \"your@email.com\"

[server]
headless = true
enableCORS=false
port = \$PORT
" > ~/.streamlit/config.toml
