import gradio as gr


def render_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except FileNotFoundError as e:
        raise e


def create_gradio_interface():
    readme_content = render_readme()
    with gr.Blocks() as demo:
        gr.Markdown(readme_content)
    return demo


# 启动 Gradio 接口
if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch()
