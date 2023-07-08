##나만의 음성인식-텍스트 변환기!..!! 1차 버전 with ChatGPT


import speech_recognition as sr

def 음성_인식():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("말씀해주세요:")
        audio = r.listen(source)

    try:
        인식된_문장 = r.recognize_google(audio, language="ko-KR")
        return 인식된_문장
    except sr.UnknownValueError:
        print("음성을 인식할 수 없습니다.")
    except sr.RequestError as e:
        print("Google Speech Recognition 서비스에 접근할 수 없습니다. {0}".format(e))

def 저장하기(파일명, 내용):
    with open(파일명, 'a', encoding='utf-8') as 파일:
        파일.write(내용 + '\n')
        print("내용이 성공적으로 저장되었습니다.")

# 음성을 인식하여 변환된 내용을 저장할 파일명을 입력받습니다.
파일명 = input("저장할 파일명을 입력하세요: ")

# 음성 인식 함수를 호출하여 사용자의 말을 인식합니다.
인식된_말 = 음성_인식()

# 변환된 말을 파일에 저장합니다.
if 인식된_말:
    저장하기(파일명, 인식된_말)
else:
    print("음성을 인식하지 못했습니다.")
