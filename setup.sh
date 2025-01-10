pip install -e ./src/model
pip install -e ./src/data

mkdir -p ~/.streamlit

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
