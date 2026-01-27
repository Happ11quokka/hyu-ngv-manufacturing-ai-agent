# Luxia API Reference

## Base URL
```
https://bridge.luxiacloud.com
```

## 인증
모든 API 요청 시 Header에 API 키 필요:
```
apikey: YOUR_API_KEY
```

---

## 1. Chat (Luxia 3)

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /luxia/v1/chat` |
| **설명** | LLM 채팅 응답 생성 |

### 사용 가능 모델
- `luxia3-llm-8b-0731`
- `luxia3-llm-32b-0731`
- `luxia3-deep-32b-0731`
- `luxia2.5-llm-32b-0505`
- `luxia2.5-deep-32b-0505`
- `luxia2.5-llm-8b-0401`
- `luxia2.5-llm-32b-0401`

### 파라미터
| 파라미터 | 타입 | 필수 | 설명 | 기본값 |
|----------|------|------|------|--------|
| `model` | string | O | 모델 ID | - |
| `messages` | array | O | 대화 메시지 배열 | - |
| `temperature` | number | X | 샘플링 온도 (0~1) | 0 |
| `max_completion_tokens` | number | X | 최대 생성 토큰 수 | 512 |
| `top_p` | number | X | Nucleus sampling (0~1) | 1 |
| `frequency_penalty` | number | X | 빈도 패널티 (0~2) | 0 |
| `stream` | boolean | X | 스트리밍 응답 여부 | false |

### 요청 예시
```bash
curl --location 'https://bridge.luxiacloud.com/luxia/v1/chat' \
  --header 'apikey: YOUR_API_KEY' \
  --header 'Content-Type: application/json' \
  -d '{
    "model": "luxia3-llm-8b-0731",
    "messages": [
      {
        "role": "user",
        "content": "안녕하세요"
      }
    ],
    "stream": true,
    "temperature": 0,
    "max_completion_tokens": 2048
  }'
```

---

## 2. Document Parse

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /luxia/v1/document-parse` |
| **설명** | PDF, Word 문서 파싱 및 텍스트/구조 추출 |

### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `file` | file | O | 파싱할 문서 (PDF, Word) |
| `output_mode` | string | O | 출력 형식: `md`, `html`, `xml` |
| `ocr` | string | X | OCR 모드: `auto`, `force` |

### 요청 예시
```bash
curl --location 'https://bridge.luxiacloud.com/luxia/v1/document-parse' \
  --header 'apikey: YOUR_API_KEY' \
  --form 'output_mode="html"' \
  --form 'file=@"/path/to/your/file.pdf"' \
  --form 'ocr="auto"'
```

---

## 3. Document AI

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /luxia/v1/document-ai` |
| **설명** | 이미지 기반 문서 분석 (멀티모달 비전) |

### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `image` | string | O | 이미지 URL 또는 base64 인코딩 데이터 |

### 요청 예시
```bash
curl -X POST 'https://bridge.luxiacloud.com/luxia/v1/document-ai' \
  -H 'Content-Type: application/json' \
  -H 'apikey: YOUR_API_KEY' \
  -d '{
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAC..."
  }'
```

---

## 4. Chunk

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /luxia/v1/document-chunk` |
| **설명** | 문서를 청크로 분할 |

### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `text` | string | O | 청킹할 문서 내용 |
| `chunk_sizes` | number | O | 청크 크기 |
| `overlap` | number | O | 청크 간 오버랩 크기 |
| `type` | string | O | 청킹 타입: `length`, `token`, `sentence` |
| `separator` | string | O | 구분자 정규식 |
| `enhanceChunkQuality` | boolean | X | 청크 품질 향상 여부 |

### 요청 예시
```bash
curl -X POST 'https://bridge.luxiacloud.com/luxia/v1/document-chunk' \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Saltlux is an AI company...",
    "chunk_sizes": 300,
    "overlap": 100,
    "type": "length",
    "separator": "chapter\\W+\\d",
    "enhanceChunkQuality": true
  }'
```

---

## 5. Rerank

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /luxia/v1/rerank` |
| **설명** | 문서 목록을 쿼리 관련성 기준으로 재정렬 |

### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `model` | string | O | 모델명 (예: `luxia-rerank-2501`) |
| `query` | string | O | 검색 쿼리 |
| `documents` | array | O | 재정렬할 문서 목록 |
| `top_k` | number | X | 반환할 최대 결과 수 |
| `lang` | string | X | 언어 (`ko` 또는 `en`) |

### 요청 예시
```bash
curl -X POST 'https://bridge.luxiacloud.com/luxia/v1/rerank' \
  -H 'Content-Type: application/json' \
  -H 'apikey: YOUR_API_KEY' \
  -d '{
    "model": "luxia-rerank-2501",
    "query": "hello, how are you",
    "documents": ["this is example", "nice to meet you"],
    "top_k": 3,
    "lang": "en"
  }'
```

---

## 6. Query Expansion

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /luxia/v1/query-expansion` |
| **설명** | 검색 쿼리를 의미적으로 확장 |

### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `query` | string | O | 원본 검색 쿼리 |
| `n` | number | O | 생성할 확장 쿼리 수 |
| `expansion_type` | string | O | 확장 타입: `basic`, `ko-creative`, `en-creative` |

### 요청 예시
```bash
curl -X POST 'https://bridge.luxiacloud.com/luxia/v1/query-expansion' \
  -H 'Content-Type: application/json' \
  -H 'apikey: YOUR_API_KEY' \
  -d '{
    "query": "Tell me about global warming",
    "n": 5,
    "expansion_type": "basic"
  }'
```

---

## 7. Triple Extraction

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /luxia/v1/triple-extraction` |
| **설명** | 텍스트에서 주어-서술어-목적어 관계 추출 |

### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `query` | string | O | 분석할 텍스트 |
| `use_unk` | boolean | O | 복잡한 구조에서 "UNK" 사용 여부 |

### 요청 예시
```bash
curl -X POST 'https://bridge.luxiacloud.com/luxia/v1/triple-extraction' \
  -H 'Content-Type: application/json' \
  -H 'apikey: YOUR_API_KEY' \
  -d '{
    "query": "Why do we have four seasons on Earth?",
    "use_unk": true
  }'
```

---

## 8. Text-to-Speech (TTS)

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /luxia/v1/text-to-speech` |
| **설명** | 텍스트를 음성으로 변환 |

### 파라미터
| 파라미터 | 타입 | 필수 | 설명 | 기본값 |
|----------|------|------|------|--------|
| `input` | string | O | 합성할 텍스트 | - |
| `voice` | integer | O | 음성 ID | - |
| `lang` | string | O | 언어 코드 (`ko`, `en`) | - |
| `stream` | boolean | X | 스트리밍 여부 | false |
| `response_format` | string | X | 오디오 형식 | wav |
| `tempo` | number | X | 속도 (0.8~1.2) | 1 |
| `pad_silence` | number | X | 끝 무음 (초) | 0 |
| `gain_db` | number | X | 볼륨 조절 (-5.0~0.0) | 0 |
| `sample_rate` | string | X | 샘플레이트 | 24k |

### 요청 예시
```bash
curl -X POST 'https://bridge.luxiacloud.com/luxia/v1/text-to-speech' \
  -H "apikey: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "안녕하세요. 반갑습니다.",
    "voice": 76,
    "lang": "ko"
  }' \
  --output output.wav
```

---

## 9. Ask Goover - Chat RAG

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /api/goover/askgoover/rag` |
| **설명** | 참조 문서 기반 RAG 채팅 (스트리밍 응답) |

### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `messages` | array | O | 대화 메시지 배열 |
| `feedIds` | array | X | 참조 문서 ID 목록 |

### 요청 예시
```bash
curl --location 'https://bridge.luxiacloud.com/api/goover/askgoover/rag' \
  --header 'apikey: YOUR_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{
    "messages": [
      {"role": "user", "content": "Summarize the document content."}
    ],
    "feedIds": ["go-public-web-kor-xxx", "go-public-web-kor-yyy"]
  }'
```

---

## 10. Report Generation

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /api/goover/briefing/report` |
| **설명** | 상세 보고서 생성 |

### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `query` | string | O | 보고서 생성 쿼리 |
| `isRag` | boolean | O | RAG 검색 사용 여부 |
| `feedIds` | array | X | 참조 문서 ID 목록 |
| `reportType` | string | X | 보고서 타입 (예: `general`) |
| `styleType` | string | X | 스타일 (예: `essay`) |

### 요청 예시
```bash
curl --location 'https://bridge.luxiacloud.com/api/goover/briefing/report' \
  --header 'apikey: YOUR_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{
    "query": "엔비디아",
    "reportType": "general",
    "styleType": "essay",
    "feedIds": ["go-public-web-kor-xxx"],
    "isRag": true
  }'
```

---

## 11. Topic Summary

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /api/goover/briefing/topic-summary` |
| **설명** | 문서별 토픽 추출 및 요약 |

### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `query` | string | O | 요약 쿼리 |
| `isRag` | boolean | O | RAG 검색 사용 여부 |
| `feedIds` | array | X | 참조 문서 ID 목록 |

### 요청 예시
```bash
curl --location 'https://bridge.luxiacloud.com/api/goover/briefing/topic-summary' \
  --header 'apikey: YOUR_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{
    "isRag": true,
    "query": "NVIDIA 주가동향 알려줘",
    "feedIds": ["go-public-news-kor-xxx"]
  }'
```

---

## 12. Connectome Extraction

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /api/goover/briefing/connectome` |
| **설명** | 지식 그래프/커넥톰 추출 |

### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `query` | string | O | 추출 쿼리 |
| `isRag` | boolean | O | RAG 검색 사용 여부 |
| `feedIds` | array | X | 참조 문서 ID 목록 |
| `language` | string | X | 언어 (`ko`, `en`) |
| `maxMainSize` | number | X | 메인 노드 최대 수 |
| `maxSubSize` | number | X | 서브 노드 최대 수 |

---

## 13. Entity Extraction (Related People & Companies)

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /api/goover/briefing/v1/relate-entity` |
| **설명** | 관련 인물 및 기업 추출 |

### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `isRag` | boolean | O | RAG 검색 사용 여부 |
| `feedIds` | array | X | 참조 문서 ID 목록 |
| `language` | string | X | 언어 (`ko`, `en`) |

---

## 14. Scrap APIs

### 14.1 Scrap Web

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /api/goover/scrap/index/web` |
| **설명** | 웹 URL 인덱싱 → feedId 반환 |

```bash
curl --location 'https://bridge.luxiacloud.com/api/goover/scrap/index/web' \
  --header 'apikey: YOUR_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{
    "urls": ["http://naver.com", "https://google.com"]
  }'
```

### 14.2 Scrap Text

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /api/goover/scrap/index/text` |
| **설명** | 텍스트 인덱싱 → feedId 반환 |

```bash
curl --location 'https://bridge.luxiacloud.com/api/goover/scrap/index/text' \
  --header 'apikey: YOUR_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{
    "text": "인덱싱할 텍스트 내용..."
  }'
```

### 14.3 Scrap File

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /api/goover/scrap/index/file` |
| **설명** | 파일 인덱싱 (txt, pdf, docx, pptx, xlsx) |

```bash
curl --location 'https://bridge.luxiacloud.com/api/goover/scrap/index/file' \
  --header 'apikey: YOUR_API_KEY' \
  --form 'files=@"./document.pdf"' \
  --form 'stream="false"'
```

### 14.4 Metasearch

| 항목 | 내용 |
|------|------|
| **Endpoint** | `GET /api/goover/scrap/metasearch` |
| **설명** | 외부 소스 검색 |

```bash
curl --location 'https://bridge.luxiacloud.com/api/goover/scrap/metasearch?keyword=검색어' \
  --header 'apikey: YOUR_API_KEY'
```

---

## 15. Search APIs

### 15.1 Search Feed

| 항목 | 내용 |
|------|------|
| **Endpoint** | `GET /api/goover/search/feed` |
| **설명** | 피드(뉴스/웹) 검색 |

```bash
curl --location 'https://bridge.luxiacloud.com/api/goover/search/feed?keyword=검색어&language=ko&offset=0&limit=10' \
  --header 'apikey: YOUR_API_KEY'
```

### 15.2 Search Report

| 항목 | 내용 |
|------|------|
| **Endpoint** | `GET /api/goover/search/report` |
| **설명** | 보고서 검색 |

### 15.3 Search Briefing

| 항목 | 내용 |
|------|------|
| **Endpoint** | `GET /api/goover/search/briefing` |
| **설명** | 브리핑 페이지 검색 |

---

## 16. Hybrid Search

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /api/goover/search/hybrid/list` |
| **설명** | 하이브리드 검색 (키워드 + 시멘틱) |

### 파라미터
| 파라미터 | 타입 | 필수 | 설명 | 기본값 |
|----------|------|------|------|--------|
| `q` | string | O | 검색 쿼리 | - |
| `n` | number | O | 최대 결과 수 | - |
| `startDate` | string | X | 검색 시작일 | 전체 |
| `endDate` | string | X | 검색 종료일 | 전체 |
| `sortBy` | string | X | 정렬 기준 | date |

### 요청 예시
```bash
curl --location 'https://bridge.luxiacloud.com/api/goover/search/hybrid/list' \
  --header 'apikey: YOUR_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{
    "q": "기후 변화",
    "n": 50,
    "startDate": "2025-04-01",
    "endDate": "2025-08-01",
    "sortBy": "score"
  }'
```

---

## 17. Diquest Detect Landmark

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /vision/ploonet/meta/v1/detect-landmark` |
| **설명** | 이미지에서 랜드마크 감지 |

### 요청 예시
```bash
curl --location 'https://bridge.luxiacloud.com/vision/ploonet/meta/v1/detect-landmark' \
  --header 'apikey: YOUR_API_KEY' \
  --form 'file=@/path/to/image.png'
```

---

## 18. OpenAI ChatGPT

### 18.1 GPT-4o Mini

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /llm/openai/chat/completions/gpt-4o-mini/create` |
| **설명** | GPT-4o Mini 채팅 |

```bash
curl --location 'https://bridge.luxiacloud.com/llm/openai/chat/completions/gpt-4o-mini/create' \
  --header 'apikey: YOUR_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### 18.2 GPT-4o

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /llm/openai/chat/completions/gpt-4o/create` |
| **설명** | GPT-4o 채팅 |

```bash
curl --location 'https://bridge.luxiacloud.com/llm/openai/chat/completions/gpt-4o/create' \
  --header 'apikey: YOUR_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

## 19. Google AI (Gemini)

### 19.1 Gemini Generate Content

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /llm/google/gemini/generate/flash20/content` |
| **설명** | Gemini Flash 2.0 텍스트 생성 |

#### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `model` | string | O | 모델명 (예: `gemini-2.0-flash`) |
| `contents` | string | O | 프롬프트 또는 사용자 콘텐츠 |

#### 요청 예시
```python
import requests

url = "https://bridge.luxiacloud.com/llm/google/gemini/generate/flash20/content"
headers = {
    "apikey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "model": "gemini-2.0-flash",
    "contents": "Tell me about Saturn"
}
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

#### 응답 예시
```json
{
  "results": [
    {
      "candidates": [
        {
          "content": {
            "parts": [{ "text": "응답 텍스트..." }],
            "role": "model"
          },
          "finishReason": "STOP",
          "avgLogprobs": -0.209
        }
      ],
      "modelVersion": "gemini-2.0-flash",
      "usageMetadata": {
        "promptTokenCount": 22,
        "candidatesTokenCount": 46,
        "totalTokenCount": 68
      }
    }
  ]
}
```

### 19.2 Gemini Multi-Turn

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /llm/google/gemini/generate/flash20/content-multi-turn` |
| **설명** | 멀티턴 대화 |

#### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `model` | string | O | 모델명 |
| `history` | array | O | 대화 히스토리 (role/parts 객체 배열) |
| `message1` | string | X | 첫 번째 후속 메시지 |
| `message2` | string | X | 두 번째 후속 메시지 |

#### 요청 예시
```bash
curl --location 'https://bridge.luxiacloud.com/llm/google/gemini/generate/flash20/content-multi-turn' \
  --header 'apikey: YOUR_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "gemini-2.0-flash",
    "history": [
      { "role": "user", "parts": [{ "text": "Hello" }] },
      { "role": "model", "parts": [{ "text": "Great to meet you." }] }
    ],
    "message1": "I have 2 dogs.",
    "message2": "How many paws are in my house?"
  }'
```

### 19.3 Gemini Stream

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /llm/google/gemini/generate/flash20/content-stream` |
| **설명** | 스트리밍 텍스트 생성 |

#### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `model` | string | O | 모델명 |
| `contents` | string | O | 프롬프트 |

#### 요청 예시
```bash
curl --location 'https://bridge.luxiacloud.com/llm/google/gemini/generate/flash20/content-stream' \
  --header 'apikey: YOUR_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "gemini-2.0-flash",
    "contents": "Tell me about the Roman Empire"
  }'
```

### 19.4 Gemini Multi-Modal (Image) - ⚠️ Coming Soon

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /llm/google/gemini/generate/flash20/content-multi-modal` |
| **설명** | 이미지 URL 포함 멀티모달 생성 |
| **상태** | **Coming Soon** - 아직 사용 불가 |

---

## 20. Google Text-to-Speech

### 20.1 TTS (Base64 응답)

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /tts/google-cloud/audio/speech` |
| **설명** | 텍스트→음성 변환 (Base64 반환) |

```bash
curl --location 'https://bridge.luxiacloud.com/tts/google-cloud/audio/speech' \
  --header 'apikey: YOUR_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{
    "text": "안녕하세요",
    "languageCode": "ko-KR",
    "voiceName": "ko-KR-Standard-A",
    "audioEncoding": "MP3"
  }'
```

### 20.2 TTS Link (파일 링크 응답)

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /tts/google-cloud/audio/speech-link` |
| **설명** | 텍스트→음성 변환 (파일 URL 반환) |

---

## 21. DeepL Translate

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /language/deepl/v2/translate/text` |
| **설명** | 텍스트 번역 |

### 요청 예시
```bash
curl --location 'https://bridge.luxiacloud.com/language/deepl/v2/translate/text' \
  --header 'apikey: YOUR_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{
    "text": "Hello, world!",
    "target_lang": "KO"
  }'
```

---

## 22. Speech-to-Text (STT)

### 22.1 GPT-4o Transcribe

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /stt/openai/gpt-4o/transcriptions` |
| **설명** | 오디오 → 텍스트 변환 |

```bash
curl --location 'https://bridge.luxiacloud.com/stt/openai/gpt-4o/transcriptions' \
  --header 'apikey: YOUR_API_KEY' \
  --form 'file=@/path/to/audio/test.wav'
```

### 22.2 Whisper-1 Timestamp

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /stt/openai/gpt-4o/whisper-1-timestamp` |
| **설명** | 타임스탬프 포함 음성 인식 |

```bash
curl --location 'https://bridge.luxiacloud.com/stt/openai/gpt-4o/whisper-1-timestamp' \
  --header 'apikey: YOUR_API_KEY' \
  --form 'file=@/path/to/audio/test.wav'
```

### 22.3 STT 16K Batch

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /stt/luxia/batch16k/speech-to-text` |
| **설명** | 배치 음성 인식 (16kHz) |

```bash
curl --location 'https://bridge.luxiacloud.com/stt/luxia/batch16k/speech-to-text' \
  --header 'apikey: YOUR_API_KEY' \
  --form 'file=@/path/to/audio/test.wav'
```

---

## API 응답 코드

| 코드 | 설명 |
|------|------|
| `0` | 성공 |
| `-1` | 실패 |

---

## 인덱싱 상태 (Scrap APIs)

| 상태 | 설명 |
|------|------|
| `READY` | 준비 완료 |
| `PARSING` | 파일에서 텍스트 추출 중 |
| `SCRAPING` | URL에서 텍스트 추출 중 |
| `CHUNKING` | 텍스트 분할 중 |
| `EMBEDDING` | 벡터 변환 중 |
| `INDEXING` | 인덱싱 중 |
| `COMPLETE` | 완료 |
