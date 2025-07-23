import argparse
from io import BytesIO
from time import sleep

import helium
import PIL.Image
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from smolagents import CodeAgent, WebSearchTool, tool
from smolagents.agents import ActionStep
from smolagents.cli import load_model


github_request = """
Tôi muốn tìm hiểu công việc cần phải làm để có thể có một repo trong github.com/trending.
Bạn có thể điều hướng đến trang cá nhân của tác giả top của repo trending nhất, và cho tôi biết tổng số lượng commit của họ trong năm qua không?
"""  # The agent is able to achieve this request only when powered by GPT-4o or Claude-3.5-sonnet.

search_request = """
Hãy điều hướng đến trang https://en.wikipedia.org/wiki/Chicago và cho tôi biết một câu chứa từ "1992" nói về một vụ tai nạn xây dựng.
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="Chạy một script tự động hóa trình duyệt web với một mô hình đã chỉ định.")
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default=search_request,
        help="Yêu cầu để chạy với agent",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="LiteLLMModel",
        help="Loại mô hình để sử dụng (ví dụ: OpenAIServerModel, LiteLLMModel, TransformersModel, InferenceClientModel)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gpt-4o",
        help="ID mô hình để sử dụng cho loại mô hình đã chỉ định",
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Nhà cung cấp dự đoán để sử dụng cho mô hình",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        help="Cơ sở API để sử dụng cho mô hình",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key để sử dụng cho mô hình",
    )
    return parser.parse_args()


def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    driver = helium.get_driver()
    current_step = memory_step.step_number
    if driver is not None:
        for previous_memory_step in agent.memory.steps:  # Remove previous screenshots from logs for lean processing
            if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
                previous_memory_step.observations_images = None
        png_bytes = driver.get_screenshot_as_png()
        image = PIL.Image.open(BytesIO(png_bytes))
        print(f"Captured a browser screenshot: {image.size} pixels")
        memory_step.observations_images = [image.copy()]  # Create a copy to ensure it persists, important!

    # Update observations with current URL
    url_info = f"Current url: {driver.current_url}"
    memory_step.observations = (
        url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
    )
    return


@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
    Args:
        text: The text to search for
        nth_result: Which occurrence to jump to (default: 1)
    """
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    if nth_result > len(elements):
        raise Exception(f"Match n°{nth_result} not found (only {len(elements)} matches found)")
    result = f"Found {len(elements)} matches for '{text}'."
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    result += f"Focused on element {nth_result} of {len(elements)}"
    return result


@tool
def go_back() -> None:
    """Goes back to previous page."""
    driver.back()


@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows! This does not work on cookie consent banners.
    """
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()


def initialize_driver():
    """Initialize the Selenium WebDriver."""
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--window-size=1000,1350")
    chrome_options.add_argument("--disable-pdf-viewer")
    chrome_options.add_argument("--window-position=0,0")
    return helium.start_chrome(headless=False, options=chrome_options)


def initialize_agent(model):
    """Initialize the CodeAgent with the specified model."""
    return CodeAgent(
        tools=[WebSearchTool(), go_back, close_popups, search_item_ctrl_f],
        model=model,
        additional_authorized_imports=["helium"],
        step_callbacks=[save_screenshot],
        max_steps=20,
        verbosity_level=2,
    )


helium_instructions = """
Sử dụng công cụ web_search của bạn khi bạn muốn nhận kết quả tìm kiếm Google.
Sau đó bạn có thể sử dụng helium để truy cập các trang web. Đừng sử dụng helium cho tìm kiếm Google, chỉ dùng cho điều hướng trang web!
Đừng lo lắng về trình điều khiển helium, nó đã được quản lý.
Chúng tôi đã chạy "from helium import *"
Sau đó bạn có thể đi tới các trang!
<code>
go_to('github.com/trending')
</code>

Bạn có thể trực tiếp nhấp vào các phần tử có thể nhấp bằng cách nhập văn bản xuất hiện trên chúng.
<code>
click("Top products")
</code>

Nếu đó là một liên kết:
<code>
click(Link("Top products"))
</code>

Nếu bạn cố gắng tương tác với một phần tử và nó không được tìm thấy, bạn sẽ gặp LookupError.
Nói chung hãy dừng hành động của bạn sau mỗi lần nhấp nút để xem điều gì xảy ra trên ảnh chụp màn hình của bạn.
Không bao giờ cố gắng đăng nhập vào một trang.

Để cuộn lên hoặc xuống, sử dụng scroll_down hoặc scroll_up với đối số là số pixel để cuộn.
<code>
scroll_down(num_pixels=1200) # Điều này sẽ cuộn xuống một khung nhìn
</code>

Khi bạn có các cửa sổ bật lên với biểu tượng chữ thập để đóng, đừng cố gắng nhấp vào biểu tượng đóng bằng cách tìm phần tử của nó hoặc nhắm mục tiêu phần tử 'X' (điều này thường thất bại).
Chỉ cần sử dụng công cụ tích hợp `close_popups` để đóng chúng:
<code>
close_popups()
</code>

Bạn có thể sử dụng .exists() để kiểm tra sự tồn tại của một phần tử. Ví dụ:
<code>
if Text('Accept cookies?').exists():
    click('I accept')
</code>

Tiến hành theo nhiều bước thay vì cố gắng giải quyết nhiệm vụ trong một lần.
Và cuối cùng, chỉ khi bạn có câu trả lời, hãy trả về câu trả lời cuối cùng của bạn.
<code>
final_answer("CÂU_TRẢ_LỜI_CỦA_BẠN_Ở_ĐÂY")
</code>

Nếu các trang có vẻ bị kẹt khi tải, bạn có thể phải chờ, ví dụ `import time` và chạy `time.sleep(5.0)`. Nhưng đừng lạm dụng điều này!
Để liệt kê các phần tử trên trang, ĐỪNG thử tìm kiếm phần tử dựa trên mã như 'contributors = find_all(S("ol > li"))': chỉ cần nhìn vào ảnh chụp màn hình mới nhất mà bạn có và đọc nó bằng mắt, hoặc sử dụng công cụ search_item_ctrl_f của bạn.
Tất nhiên, bạn có thể hành động trên các nút như người dùng sẽ làm khi điều hướng.
Sau mỗi khối mã bạn viết, bạn sẽ tự động được cung cấp ảnh chụp màn hình cập nhật của trình duyệt và url trình duyệt hiện tại.
Nhưng hãy cẩn thận rằng ảnh chụp màn hình sẽ chỉ được chụp ở cuối toàn bộ hành động, nó sẽ không thấy các trạng thái trung gian.
Đừng tắt trình duyệt.
Khi bạn có các cửa sổ modal hoặc banner cookie trên màn hình, bạn nên loại bỏ chúng trước khi có thể nhấp vào bất cứ thứ gì khác.
"""


def run_webagent(
    prompt: str,
    model_type: str,
    model_id: str,
    provider: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> None:
    # Load environment variables
    load_dotenv()

    # Initialize the model based on the provided arguments
    model = load_model(model_type, model_id, provider=provider, api_base=api_base, api_key=api_key)

    global driver
    driver = initialize_driver()
    agent = initialize_agent(model)

    # Run the agent with the provided prompt
    agent.python_executor("from helium import *")
    agent.run(prompt + helium_instructions)


def main() -> None:
    # Parse command line arguments
    args = parse_arguments()
    run_webagent(args.prompt, args.model_type, args.model_id, args.provider, args.api_base, args.api_key)


if __name__ == "__main__":
    main()
