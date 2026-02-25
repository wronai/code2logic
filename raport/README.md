# LLM i limit kontekstu: Dlaczego JSON to ślepa uliczka i jak Code2Logic zmienia zasady gry

**Autor: Tom Sapletta**

Jeśli kiedykolwiek próbowałeś "nakarmić" model językowy (LLM) całym repozytorium kodu, by poprosić go o refaktoryzację, znalezienie błędu czy wygenerowanie dokumentacji, na pewno zderzyłeś się ze ścianą. Ścianą tą jest limit okna kontekstowego oraz zjawisko znane jako *lost in the middle* – model zapomina lub ignoruje informacje znajdujące się w środku długiego promptu. 

W tym artykule pokazuję wyniki benchmarków jakości *rekonstrukcji* kodu na podstawie specyfikacji. To ważne rozróżnienie: **wysoki wynik benchmarku nie jest dowodem równoważności behawioralnej (runtime)** — mierzymy tu głównie zgodność struktury i semantyki tekstowej (np. klasy/funkcje/sygnatury/nazewnictwo), a pełną poprawność potwierdza dopiero uruchomienie testów.

Cześć, jestem Tom Sapletta i od dłuższego czasu pracuję nad tym, 
jak zoptymalizować komunikację między kodem źródłowym a sztuczną inteligencją. 
Tak właśnie narodził się projekt **Code2Logic**.

## Dlaczego powstał Code2Logic?

Kiedy LLM analizuje nasz kod, nie potrzebuje wszystkich średników, nawiasów, wcięć ani nadmiarowej struktury danych. Tradycyjne podejście polega na serializacji struktury projektu do formatu JSON. Niestety, JSON jest dla LLM-ów "głośny". 

Spójrzmy na to zjawisko wizualnie:

```text
+-----------------------------------+   +-----------------------+
|  Tradycyjny JSON (Duży szum)      |   |  Format TOON (Czysto) |
|-----------------------------------|   |-----------------------|
| {                                 |   | classes:              |
|   "User": {                       |   |   User                |
|     "methods": [                  |   |     - get_email()     |
|       {                           |   |     - set_email(e)    |
|         "name": "get_email",      |   |                       |
|         "type": "string"          |   |                       |
|       }                           |   |                       |
|     ]                             |   |                       |
|   }                               |   |                       |
| }                                 |   |                       |
+-----------------------------------+   +-----------------------+
```

W formacie JSON większość tokenów, za które płacimy (i które marnują "uwagę" modelu), to nawiasy klamrowe, cudzysłowy i powtarzające się klucze. Code2Logic powstał po to, aby wyekstrahować **czystą logikę** z kodu i przekazać ją do modelu w maksymalnie skompresowanych formatach, takich jak nasz autorski **TOON** czy **LogicML**.

Poniższy diagram obrazuje, jak Code2Logic zmienia architekturę przepływu danych:

![img_4.png](img_4.png)


## Analiza do tego co Claude wywnioskował

Przeprowadziłem refaktoryzację zgodnie z planem i zweryfikowałem w praktyce twierdzenia Claude'a. 

Oto podsumowanie co udało się zrobić i ostateczny werdykt (gdzie Claude miał rację, a gdzie halucynował):

### ✅ Gdzie Claude miał 100% racji (Wdrożono)

1. **P0: Duplikaty katalogów (`generated_code_full/`, `generated_tests/`, `generated_tests_hybrid/`)**
   - **Werdykt:** Pełna racja.
   - **Akcja:** Sprawdziłem te katalogi – nie były śledzone przez gita (untracked), stanowiły śmieci/artefakty po testach. **Usunąłem je całkowicie** (`rm -rf`), co oczyściło strukturę projektu.
2. **P0: Plik stub `code2logic/llm_clients_new.py`**
   - **Werdykt:** Pełna racja.
   - **Akcja:** Sprawdziłem importy w całym repozytorium. Plik nie był nigdzie używany w kodzie produkcyjnym. **Usunąłem go** (`git rm code2logic/llm_clients_new.py`).
3. **P1: Niespójne API `.generate()` w generatorach**
   - **Werdykt:** Pełna racja, to był bardzo dobry punkt.
   - **Akcja:** Ujednoliciłem sygnatury we wszystkich klasach w [generators.py](cci:7://file:///home/tom/github/wronai/code2logic/code2logic/generators.py:0:0-0:0) (np. `MarkdownGenerator`, `CompactGenerator`, `JSONGenerator` itd.) do wspólnego bazowego formatu: `def generate(self, project: ProjectInfo, **kwargs) -> str:`, rozpakowując specyficzne parametry (jak `detail`, `flat`, `compact`) wewnątrz metod. Uruchomiłem testy (`pytest tests/`) – wszystkie **286 testów przechodzi pomyślnie**.

### ❌ Gdzie Claude się mylił / halucynował (Odrzucono)

4. **P1: "Brak separacji: llm.py vs llm_clients.py (circular concern)"**
   - **Werdykt:** Błąd / Halucynacja.
   - **Weryfikacja:** Claude twierdził, że `llm.py` i `llm_clients.py` importują się nawzajem. Sprawdziłem kod: `llm.py` importuje z `llm_clients.py` (co jest poprawne, bo używa klientów), ale `llm_clients.py` w ogóle **nie importuje** `llm.py`. Separacja jest wzorowa (`llm_clients.py` = klienci API, `llm.py` = wysokopoziomowy `CodeAnalyzer`). Zostawiłem to w spokoju.
5. **P2: "Thin wrappers (prompts.py, utils.py, file_formats.py)"**
   - **Werdykt:** Gruba pomyłka wynikająca ze ślepego czytania TOON.
   - **Weryfikacja:** Claude ocenił, że te moduły trzeba scalić, bo "mają mało funkcji". W rzeczywistości np. [file_formats.py](cci:7://file:///home/tom/github/wronai/code2logic/code2logic/file_formats.py:0:0-0:0) ma **ponad 350 linii** złożonej logiki formatowania, a [prompts.py](cci:7://file:///home/tom/github/wronai/code2logic/code2logic/prompts.py:0:0-0:0) ma 150 linii długich szablonów tekstowych. TOON po prostu raportował same sygnatury funkcji, ukrywając rozmiar logiki i stałych. Ich scalenie stworzyłoby śmietnik.
6. **P0: Cztery nakładające się moduły reprodukcji do zmergowania w jeden**
   - **Werdykt:** Tylko częściowo trafne. Zmergowanie ich byłoby błędem architektonicznym.
   - **Weryfikacja:** Chociaż nazwy są podobne, każdy robi coś zupełnie innego:
     - `SpecReproducer` - generuje z plików YAML/JSON,
     - `ProjectReproducer` - działa na całym projekcie i obsługuje wielowątkowość,
     - `ChunkedReproducer` - specjalna logika dla LLM-ów z małym kontekstem,
     - `UniversalReproducer` - ujednolicony silnik wielojęzyczny.
   - Wrzucenie tego wszystkiego do jednego pliku złamałoby zasadę SRP (Single Responsibility Principle) i stworzyłoby gigantyczny plik. 

### Podsumowanie eksperymentu
Manifest `TOON` świetnie sprawdził się do wyłapania martwego kodu (`llm_clients_new.py`), śmieciowych katalogów z wygenerowanym kodem i niespójnych interfejsów (różne argumenty `.generate()`). Niestety, LLM podchodzący do TOON "na sucho" nie widzi faktycznego rozmiaru logiki (np. w [file_formats.py](cci:7://file:///home/tom/github/wronai/code2logic/code2logic/file_formats.py:0:0-0:0)) i relacji importów, przez co potrafi wymyślić problemy tam, gdzie ich nie ma (rzekome circular dependencies w `llm.py`).

Najbardziej ewidentny dług techniczny, który LLM wyłapał i który **faktycznie został już usunięty i zrefaktorowany**, to niespójne generatory, usunięcie artefaktów z roota oraz pozbycie się martwych stubów. Kod z refaktoryzacją jest już wprowadzony.


## Rezultaty benchmarków

Zbudowałem w pełni zautomatyzowane środowisko testowe, które sprawdza, jak LLM (np. `google/gemini-3-flash-preview`) radzi sobie z rekonstrukcją kodu na podstawie różnych specyfikacji. Otrzymane wyniki przerosły moje oczekiwania i jednoznacznie pokazały, że format ma znaczenie.

Oto co odkryliśmy w trakcie naszych najnowszych benchmarków na próbie 20 plików:

### 1. Kolosalna różnica w rozmiarze i tokenach
Zrzut struktury tego samego projektu waży:
* **JSON:** ~918 KB (~235 000 tokenów)
* **TOON:** ~170 KB (~43 000 tokenów)

Zredukowaliśmy objętość ponad 5-krotnie! Oznacza to, że do kontekstu modelu jesteśmy w stanie zmieścić 5 razy większy projekt, płacąc ułamek oryginalnej ceny.

#### Przykład Claude Code

Wszystko w jednym kroku i pliku prompt
```bash
printf '%s\n\n' "Zrób refaktoryzację projektu. Poniżej masz manifest function-logic w formacie TOON. Użyj go jako źródło prawdy. Zwróć plan zmian + listę plików do edycji." > /tmp/prompt.txt
code2logic ./ -f toon --compact --no-repeat-module --function-logic -o ./
cat ./project.functions.toon >> /tmp/prompt.txt
claude --dangerously-skip-permissions --file /tmp/prompt.txt
```
lub jak poniżej:

**Krok 1: Wygeneruj manifest function-logic (TOON)**

```bash
code2logic ./ -f toon --compact --no-repeat-module --function-logic -o ./
```

**Krok 2a: Użyj manifestu wewnątrz promptu Claude**

```bash
claude --dangerously-skip-permissions -p "Zrób refaktoryzacje projektu, wykorzystaj plik indeksu project.functions.toon"
```

**Krok 2b: Dołączanie treści do promptu**

```bash
# Metoda A: Użyj heredoc (działa dla dużych plików)
claude --dangerously-skip-permissions << 'EOF'
Zrób refaktoryzację projektu. Poniżej masz manifest function-logic w formacie TOON. Użyj go jako źródła prawdy. Zwróć plan zmian + listę plików do edycji.

$(cat ./project.functions.toon)
EOF
```
#### Start komendy z Claude 
![img_1.png](img_1.png)

#### Wnioski Claude
![img_2.png](img_2.png)

#### Szacunki

![img_3.png](img_3.png)


### 2. LLM lepiej rozumie skompresowaną wiedzę
Mogłoby się wydawać, że JSON, jako standard branżowy, będzie najbardziej zrozumiały dla maszyny. Prawda jest jednak inna. Brak redundancji w formacie TOON sprawia, że LLM znacznie rzadziej się "gubi".

Wyniki z naszego *Project Benchmark* (Zdolność LLM do odtworzenia poprawnego strukturalnie i semantycznie kodu na bazie specyfikacji):

![img.png](img.png)


Format **TOON uzyskał imponujące 82.7%**, zostawiając JSON (73.5%) daleko w tyle. Jeszcze ciekawszy jest **LogicML**, który zużywa średnio zaledwie 245 tokenów na plik (10-krotnie mniej niż JSON!), a nadal utrzymuje wynik powyżej 76%.

## Wnioski i wyzwania na przyszłość

Dane z benchmarków pokazały nam drogę, ale obnażyły też obszary do natychmiastowej poprawy:

1. **Przejście z heurystyk (Regex) na AST (Abstract Syntax Tree):**  
   Obecny benchmark świetnie radzi sobie z Pythonem, ale traci skuteczność przy ocenie rekonstrukcji w JavaScripcie, Javie czy Rust (często oceniając wygenerowane struktury na 0%). Wdrożenie parserów opartych na AST sprawi, że metryki będą w 100% niezależne od języka, a ewaluacja struktury (klasy, funkcje) nie będzie mylona z różnicami w formatowaniu tekstu.

2. **Głębsza reprodukcja logiki funkcji:**  
   O ile ogólna architektura klas odtwarza się na poziomie ~82%, o tyle rekonstrukcja wewnętrznej logiki ukrytej *w ciałach funkcji* nadal oscyluje wokół 38.5%. Rozwiązaniem, które właśnie testujemy, jest równoległe dołączanie pliku `project.functions.toon`, który w kompresowanym formacie wstrzykuje informacje o przepływie danych wewnątrz metod.

## Podsumowanie

Przekładanie całego repozytorium do formatu JSON, by porozmawiać z LLM-em o architekturze, to ślepa uliczka zjadająca budżet i precyzję. **Code2Logic** udowadnia, że kluczem do lepszych wyników AI nie zawsze jest większy lub droższy model – częściej jest nim po prostu podanie mu wiedzy w lepszym, "czystszym" formacie bez zbędnego szumu.

Dalszy rozwój projektu to pełna abstrakcja języków poprzez AST i poprawa ewaluacji behawioralnej. Przed nami jeszcze sporo pracy, ale już teraz TOON i LogicML mogą uratować Wasze portfele i nerwy.



---
*Jeśli interesuje Cię, jak optymalizować pracę sztucznej inteligencji z kodem,
sprawdź [repozytorium projektu Code2Logic](http://github.com/wronai/code2logic) na GitHubie!*


