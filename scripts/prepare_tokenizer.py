from alignment import H4ArgumentParser, PrepareTokenizerArguments
from transformers import AutoTokenizer

templates = {
    "Mistral-7B-Instruct-v0.1": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
}

def main():
    parser = H4ArgumentParser((PrepareTokenizerArguments))
    args = parser.parse()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    tokenizer.chat_template = templates[args.template]

    chat = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "I'd like to show off how chat templating works!"},
    ]

    text = tokenizer.apply_chat_template(chat, tokenize=False)
    print("Text:", text)

    if args.save_pretrained:
        tokenizer.save_pretrained(args.save_pretrained)

    if args.push_to_hub:
        assert args.hf_repo_id is not None, "Must provide --hf-repo-url if --push-to-hub is set"
        tokenizer.push_to_hub(args.hf_repo_id, private=not args.hf_public)

if __name__ == "__main__":
    main()