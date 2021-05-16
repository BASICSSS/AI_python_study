import requests
import re

url = "https://www.billboard.com/charts/hot-100"
receive = requests.get(url)
text = receive.text


nums = re.findall(r'<span class="chart-element__rank__number">([0-9]+)', text)
names = re.findall(
    r'<span class="chart-element__information__song text--truncate color--primary">(.+)</span>',
    text,
)


# for i in zip(nums, names):
#     print(*i)

for i, (rank, title) in enumerate(zip(nums, names)):
    print("빌보드 순위 : {} / 제목 : {} ".format(rank, title))


# from bs4 import BeautifulSoup

# url = "https://www.billboard.com/charts/hot-100"
# receive = requests.get(url)
# text = receive.text

# soup = BeautifulSoup(text, "html.parser")

# # rank = soup.find("span", class_="chart-element__rank__number").get_text() #html태그를 이용한 ㅂ아식
# # rank = soup.select_one(".chart-element__rank__number").get_text()  # css를 통한 방식
# rank = soup.select(".chart-element__rank__number")
# for i in range(len(rank)):
#     rank_text = rank[i].text
#     print(rank_text)

