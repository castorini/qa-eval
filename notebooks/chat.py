import argparse
import os

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(page_title="Chat with LLMs", page_icon="🧘", layout="wide")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="HuggingFaceH4/zephyr-7b-beta",
    help="Model name",
)
args = parser.parse_args()

parser.add_argument(
    "--max_tokens",
    type=int,
    default=256,
    help="Maximum number of tokens to generate (used in OpenAI API)",
)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
TOPK = 5


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.use_default_system_prompt = False
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", low_cpu_mem_usage=True)
    return tokenizer, model


tokenizer, model = load_model()

st.subheader("Inference")
st.markdown(
    """
    <script src=" https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.min.js"></script>
    <link href=" https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    max_new_tokens = st.slider("Max new tokens", min_value=50, max_value=512, value=256)
    num_return_sequences = st.slider("Num return sequences", min_value=1, max_value=11, value=1)
    do_sample = st.checkbox("Sampling?", value=True)

    top_p = 1.0
    num_beams = 1
    if do_sample:
        top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
    else:
        num_beams = st.slider("Beam width", min_value=1, max_value=100, value=1)

    if do_sample:
        st.write(f"Nucleus sampling with top-p: {top_p}")
    elif num_beams > 1:
        st.write(f"Beam search with {num_beams} beams")
    else:
        st.write("Greedy decoding")


def respond(msg: str, system_msg: str):
    chat = []
    if system_msg:
        chat.append(
            {
                "role": "system",
                "content": system_msg,
            }
        )

    chat.extend(
        [
            {
                "role": "user" if h["is_user"] else "assistant",
                "content": h["message"] if isinstance(h["message"], str) else h["message"[0]],
            }
            for h in st.session_state.history
        ]
    )

    chat.append(
        {
            "role": "user",
            "content": msg,
        }
    )

    # while len(st.session_state.history) > 8:
    #     st.session_state.history.pop(0)
    #     st.session_state.history.pop(0)

    input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt")
    for key in inputs.keys():
        inputs[key] = inputs[key].to("cuda")

    seq_length = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )

    output = output.reshape(num_return_sequences, -1)

    resps = []
    for s in range(num_return_sequences):
        _ids = output[s]
        if not model.config.is_encoder_decoder:
            _ids = _ids[seq_length:]

        resps.append(tokenizer.decode(_ids, skip_special_tokens=True).strip())

    st.session_state.history.append({"message": msg, "is_user": True})
    st.session_state.history.append({"message": resps, "is_user": False})

    return resps


user_template = """
        <div class="card border-warning col-md-12">
            <p class="card-text text-secondary mb-1"><span class="fs-4">🧑‍💻</span> {msg}</p>
        </div>
"""

bot_template = """
        <div class="card text-bg-light border-info col-md-12">
            <p class="card-text mb-1"><span class="fs-4">🤖</span> {msg}</p>
        </div>
"""

if "history" not in st.session_state:
    st.session_state.history = []

system_col, chat_col = st.columns((0.7, 1))


def clear_history():
    st.session_state.history = []
    clear_msg()


def clear_msg():
    st.session_state.user_message = ""

conversation_container = st.container()
system_msg = st.text_area("Enter system message", key="system_msg", on_change=clear_history)

msg = st.text_area("Enter your message", value="", key="user_message")
if msg:
    with st.spinner("Running..."):
        resps = respond(msg, system_msg)

    for chat in st.session_state.history:
        if isinstance(chat["message"], str):
            messages = chat["message"]
        else:
            messages = "<hr/>\n".join(chat["message"])

        conversation_container.write(
            (user_template if chat["is_user"] else bot_template).format(msg=messages),
            unsafe_allow_html=True,
        )