/** UI localization (DE / EN / JA / ZH).
 *
 * Scope rule: only *interface chrome* is translated. Domain content that comes
 * from the server or from the agent is NEVER translated here:
 *   - dataset display_name / dataset / modality / annotations / label_categories
 *     (they come from datasets.yaml and are proper nouns — "SIIM-ACR
 *     Pneumothorax", "Melanoma (MEL)" stay as-is in every language)
 *   - anything the agent writes in the transcript
 *   - the product name "Clinical-Automata"
 */
import { createContext, useContext } from "react";

// Set VITE_CONTACT_EMAIL in src/config.env.
const CONTACT_EMAIL: string = import.meta.env.VITE_CONTACT_EMAIL ?? "your-email@example.com";

export type Lang = "en" | "de" | "ja" | "zh";

export const LANGS: { code: Lang; label: string }[] = [
  { code: "en", label: "English" },
  { code: "de", label: "Deutsch" },
  { code: "ja", label: "日本語" },
  { code: "zh", label: "中文" },
];

/** The login page always opens in English; the UI only switches once the
 *  visitor picks a language. Nothing is remembered between visits — each
 *  session starts from the same neutral state. */
export const DEFAULT_LANG: Lang = "en";

const en = {
  tagline: "Clinician-driven AI development — interactive demo",
  accessCode: "Access code",
  accessCodePh: "Your access code",
  email: "Email",
  emailPh: "Where we'll email you when your model is ready",
  language: "Interface language",
  enter: "Enter",
  badCode: "Incorrect access code.",

  subtitle: "Describe the model you want. An autonomous agent builds it. No coding.",
  endSession: "End session",
  completeBtn: "Complete",
  stateActive: "active",
  stateCompleted: "completed",
  stateInterrupted: "interrupted",

  welcome:
    "✻ Welcome to the Clinical-Automata assistant (Claude Code CLI).\n\n" +
    "This is a demonstration. Click “Start using” to begin a real session.\n" +
    "You will be able to describe, in plain language, the clinical imaging\n" +
    "model you want — and an autonomous coding agent will build it for you.\n\n" +
    "No programming required. Nothing runs until you start.",
  startUsing: "Start using",
  occupied: "This demo is currently in use by another visitor. Please try again later.",

  howItWorks: "How it works",
  steps: [
    {
      title: "Explore the datasets",
      body: "On the left you'll find the available datasets. Click any card to see what annotations it provides and which label categories they cover.",
    },
    {
      title: "Warm up (optional)",
      body: "New to this kind of assistant? Feel free to chat freely with it in the middle panel first — ask anything you're curious about.",
    },
    {
      title: "Describe the model you want",
      body: "When you're ready, type what you'd like to build in the “Task description” box on the left and submit it. Plain clinical language is fine — no coding needed.",
    },
    {
      title: "Let it work — you can leave",
      body: "Setup takes a while, and the optimization that follows runs on its own. You can safely close the page — we'll email you when your model is finished.",
    },
  ],

  datasetsTitle: "Available datasets",
  dsDataset: "Dataset",
  dsModality: "Modality",
  dsImages: "Images",
  dsLicense: "License",
  dsAnnotations: "Annotations provided",
  dsCategories: "Label categories",
  dsSource: "Source ↗",

  taskLabel: "Task description",
  taskPh:
    "Describe, in your own words, the clinical AI model you'd like to build — " +
    "the task, the data, and any clinical priorities or concerns.\n\n" +
    "Not sure how to phrase it? See the example requests below.",
  taskSubmit: "Submit task & start experiment",
  taskSetupWarn:
    "Heads-up: after you submit, the setup phase can sometimes take up to 10 minutes before you see a reply. That is normal. " +
    "If there is still no reply after 20 minutes, there may be a network problem — please contact " + CONTACT_EMAIL + ".",
  taskSubmittedLabel: "Your task (submitted)",
  taskSubmittedFallback: "(submitted)",
  taskSending: "Submitting…",
  taskSendFailed: "Could not send — the connection dropped. Please try again.",
  reconnecting: "Reconnecting…",

  examplesTitle: "Example requests",
  copyRequest: "Copy request",
  copied: "Copied ✓",
  exampleModalHint:
    "This is only an example — paste it into the task box above and edit it, or write your own.",
  examples: [
    {
      title: "Pneumothorax — avoid shortcut learning",
      hint: "Chest X-ray · classification · confounder",
      text:
        "I would like to develop a classification model for pneumothorax on chest radiographs. " +
        "A major concern is that the presence of a chest drain may act as a strong confounder — " +
        "naive models may achieve apparently good performance simply by detecting chest drains, " +
        "rather than recognizing the imaging features of pneumothorax itself. Please take this " +
        "shortcut-learning issue seriously and use appropriate strategies to reduce reliance on " +
        "chest drains and encourage genuine pneumothorax feature learning.",
    },
    {
      title: "Lung lobes & zones — for teaching",
      hint: "Chest X-ray · segmentation · teaching",
      text:
        "Could you provide a lung lobe & zone segmentation model for chest X-ray teaching? " +
        "It would be helpful if the tool could switch between lung zones and anatomical lung lobes, " +
        "so that students can understand that these are not the same concept. Since lung lobes " +
        "overlap on frontal chest radiographs, the model should also show the degree of confidence " +
        "or uncertainty on the resulting output rather than presenting every boundary as definite. " +
        "Separate overlays with adjustable transparency would be ideal for teaching.",
    },
    {
      title: "Skin lesions — don't miss melanoma",
      hint: "Dermoscopy · classification · sensitivity",
      text:
        "I want an AI model to help diagnose skin lesions from dermoscopy images. Priority should be " +
        "given to melanoma: please try your best not to miss it, even if that costs some specificity. " +
        "Report melanoma sensitivity explicitly.",
    },
    {
      title: "Chest X-ray second reader — the easily-missed regions",
      hint: "Chest X-ray · detection · safety net",
      text:
        "On frontal chest radiographs, some abnormalities hide in regions that are hard to read — the lung apices, " +
        "the retrocardiac area, the costophrenic angles and the area near the diaphragm — where overlapping normal " +
        "anatomy makes them easy to overlook. I would like a tool that acts as a second reader after my own initial " +
        "read, re-checking exactly these easily-missed regions. I don't need it to re-list a full differential for the " +
        "whole film; what helps me is a reminder when something in these regions is genuinely worth a second look, " +
        "especially findings that may be obscured by ribs, clavicles, the cardiac shadow or the diaphragm. Any prompt " +
        "should be short and specific — for example 'take another look at the right apex' or 're-check the left " +
        "costophrenic angle' — not a long AI-generated explanation. I would rather receive a few well-localized prompts " +
        "that are truly worth my attention. Please make sure normal overlapping anatomy does not become a source of many " +
        "false alarms: as a last safety check it should distinguish real abnormalities from normal overlap, and if it is " +
        "unsure, I would prefer it to stay quiet rather than keep sending me back to normal regions. " +
        "(If needed, you may raise the per-round time budget from 30 min → 40 min.)",
    },
  ],

  bannerSetupTitle: "Setting up your experiment.",
  bannerSetupBody:
    "This first phase can sometimes take up to 10 minutes with little or no visible response — that is completely normal, so please be patient. " +
    "The agent may also ask you to confirm its plan before it starts the iterative optimization, so please keep an eye on this page and don't miss the question. " +
    "Once the optimization begins, no further input from you is needed: you can safely close this page and leave. " +
    "We will email you when your model is ready. " +
    "If there is still no reply after 20 minutes, there may be a network problem — please contact " + CONTACT_EMAIL + ".",
  bannerOptTitle: "Iterative optimization is running.",
  bannerOptBody:
    "No further input is needed from you. You can safely close this page and leave — the run continues on the server, " +
    "and we will email you when your model is ready.",

  transcriptHello:
    "Say hello and get a feel for the assistant. When you're ready, submit your task on the left — note that you can submit it only once.",
  agentBusy: "agent is working…",
  typePh: "Type a message…",
  waitingPh: "Waiting for the assistant…",
  send: "Send",
  chatEncourage:
    "Not sure about something? Ask anything, anytime — the assistant is always here to help you understand.",
  manualTitle: "Usage guidance",
  agentActivity: "agent activity",
  stepCount: (n: number) => `${n} step${n === 1 ? "" : "s"}`,

  runTraining: "training",
  runPseudo: "pseudo-GPU (framework mode)",
  runRunning: "running…",
  runCompleted: "completed",
  runBest: "best",
  runIteration: "iteration",
  runEpoch: "epoch",
};

export type Dict = typeof en;

const de: Dict = {
  tagline: "Klinikergesteuerte KI-Entwicklung — interaktive Demo",
  accessCode: "Zugangscode",
  accessCodePh: "Ihr Zugangscode",
  email: "E-Mail",
  emailPh: "Hierhin schicken wir die Nachricht, sobald Ihr Modell fertig ist",
  language: "Sprache der Oberfläche",
  enter: "Weiter",
  badCode: "Ungültiger Zugangscode.",

  subtitle: "Beschreiben Sie das gewünschte Modell. Ein autonomer Agent baut es. Ohne Programmieren.",
  endSession: "Sitzung beenden",
  completeBtn: "Abschließen",
  stateActive: "aktiv",
  stateCompleted: "abgeschlossen",
  stateInterrupted: "unterbrochen",

  welcome:
    "✻ Willkommen beim Clinical-Automata-Assistenten (Claude Code CLI).\n\n" +
    "Dies ist eine Demonstration. Klicken Sie auf „Starten“, um eine echte\n" +
    "Sitzung zu beginnen. Sie können in normaler Sprache beschreiben, welches\n" +
    "klinische Bildgebungsmodell Sie möchten — ein autonomer Coding-Agent\n" +
    "baut es für Sie.\n\n" +
    "Keine Programmierkenntnisse nötig. Vor dem Start läuft nichts.",
  startUsing: "Starten",
  occupied: "Diese Demo wird gerade von einer anderen Person genutzt. Bitte versuchen Sie es später erneut.",

  howItWorks: "So funktioniert es",
  steps: [
    {
      title: "Datensätze ansehen",
      body: "Links finden Sie die verfügbaren Datensätze. Klicken Sie auf eine Karte, um zu sehen, welche Annotationen sie enthält und welche Label-Kategorien abgedeckt sind.",
    },
    {
      title: "Aufwärmen (optional)",
      body: "Noch keine Erfahrung mit solchen Assistenten? Unterhalten Sie sich zunächst frei im mittleren Bereich — fragen Sie alles, was Sie interessiert.",
    },
    {
      title: "Beschreiben Sie das gewünschte Modell",
      body: "Wenn Sie bereit sind, schreiben Sie links im Feld „Aufgabenbeschreibung“, was gebaut werden soll, und senden Sie es ab. Normale klinische Sprache genügt — kein Code nötig.",
    },
    {
      title: "Lassen Sie es arbeiten — Sie können gehen",
      body: "Die Einrichtung dauert eine Weile, und die anschließende Optimierung läuft von allein. Sie können die Seite bedenkenlos schließen — wir benachrichtigen Sie per E-Mail, sobald Ihr Modell fertig ist.",
    },
  ],

  datasetsTitle: "Verfügbare Datensätze",
  dsDataset: "Datensatz",
  dsModality: "Modalität",
  dsImages: "Bilder",
  dsLicense: "Lizenz",
  dsAnnotations: "Enthaltene Annotationen",
  dsCategories: "Label-Kategorien",
  dsSource: "Quelle ↗",

  taskLabel: "Aufgabenbeschreibung",
  taskPh:
    "Beschreiben Sie in eigenen Worten das klinische KI-Modell, das Sie bauen möchten — " +
    "die Aufgabe, die Daten und alle klinischen Prioritäten oder Bedenken.\n\n" +
    "Unsicher bei der Formulierung? Sehen Sie sich unten die Beispielanfragen an.",
  taskSubmit: "Aufgabe absenden & Experiment starten",
  taskSetupWarn:
    "Hinweis: Nach dem Absenden kann die Einrichtungsphase manchmal bis zu 10 Minuten dauern, bevor eine Antwort erscheint. Das ist normal. " +
    "Wenn nach 20 Minuten immer noch keine Antwort erscheint, liegt möglicherweise ein Netzwerkproblem vor — bitte kontaktieren Sie " + CONTACT_EMAIL + ".",
  taskSubmittedLabel: "Ihre Aufgabe (abgesendet)",
  taskSubmittedFallback: "(abgesendet)",
  taskSending: "Wird gesendet…",
  taskSendFailed: "Konnte nicht gesendet werden — die Verbindung wurde unterbrochen. Bitte erneut versuchen.",
  reconnecting: "Verbindung wird wiederhergestellt…",

  examplesTitle: "Beispielanfragen",
  copyRequest: "Anfrage kopieren",
  copied: "Kopiert ✓",
  exampleModalHint:
    "Dies ist nur ein Beispiel — fügen Sie es oben in das Aufgabenfeld ein und passen Sie es an, oder schreiben Sie Ihre eigene Anfrage.",
  examples: [
    {
      title: "Pneumothorax — Shortcut-Learning vermeiden",
      hint: "Thorax-Röntgen · Klassifikation · Confounder",
      text:
        "Ich möchte ein Klassifikationsmodell für Pneumothorax auf Thorax-Röntgenaufnahmen entwickeln. " +
        "Eine große Sorge ist, dass eine liegende Thoraxdrainage als starker Confounder wirken kann — " +
        "naive Modelle erreichen scheinbar gute Ergebnisse, indem sie einfach die Drainage erkennen, " +
        "statt die bildgebenden Merkmale des Pneumothorax selbst. Bitte nehmen Sie dieses Shortcut-Learning-Problem " +
        "ernst und setzen Sie geeignete Strategien ein, um die Abhängigkeit von Thoraxdrainagen zu verringern " +
        "und das Lernen echter Pneumothorax-Merkmale zu fördern.",
    },
    {
      title: "Lungenlappen & -zonen — für die Lehre",
      hint: "Thorax-Röntgen · Segmentierung · Lehre",
      text:
        "Könnten Sie ein Segmentierungsmodell für Lungenlappen und Lungenzonen für die Lehre am Thorax-Röntgenbild bereitstellen? " +
        "Hilfreich wäre, wenn das Werkzeug zwischen Lungenzonen und anatomischen Lungenlappen umschalten könnte, " +
        "damit Studierende verstehen, dass dies nicht dasselbe ist. Da sich Lungenlappen in der frontalen Aufnahme " +
        "überlagern, sollte das Modell zusätzlich den Grad der Konfidenz bzw. Unsicherheit anzeigen, statt jede Grenze " +
        "als sicher darzustellen. Getrennte Overlays mit einstellbarer Transparenz wären für die Lehre ideal.",
    },
    {
      title: "Hautläsionen — kein Melanom übersehen",
      hint: "Dermatoskopie · Klassifikation · Sensitivität",
      text:
        "Ich möchte ein KI-Modell zur Diagnose von Hautläsionen aus dermatoskopischen Bildern. Priorität hat das Melanom: " +
        "Bitte versuchen Sie, es keinesfalls zu übersehen, auch wenn das etwas Spezifität kostet. " +
        "Geben Sie die Melanom-Sensitivität explizit an.",
    },
    {
      title: "Zweitbefunder für das Thorax-Röntgen — die leicht übersehenen Regionen",
      hint: "Thorax-Röntgen · Detektion · Sicherheitsnetz",
      text:
        "Auf frontalen Thorax-Röntgenaufnahmen verbergen sich manche Auffälligkeiten in schwer einsehbaren Regionen — " +
        "den Lungenspitzen, dem Retrokardialraum, den Randwinkeln (Recessus costophrenici) und dem Bereich nahe dem " +
        "Zwerchfell —, wo überlagernde normale Anatomie sie leicht übersehen lässt. Ich hätte gern ein Werkzeug, das nach " +
        "meiner eigenen Erstbefundung als Zweitbefunder genau diese leicht zu übersehenden Regionen noch einmal prüft. " +
        "Es soll nicht für das gesamte Bild eine vollständige Differenzialdiagnose auflisten; hilfreich ist mir ein " +
        "Hinweis, wenn in diesen Regionen wirklich etwas einen zweiten Blick wert ist — besonders Befunde, die von Rippen, " +
        "Klavikeln, dem Herzschatten oder dem Zwerchfell verdeckt sein könnten. Ein Hinweis soll kurz und konkret sein — " +
        "etwa „rechte Lungenspitze noch einmal ansehen“ oder „linken Randwinkel erneut prüfen“ — und keine lange " +
        "KI-generierte Erklärung. Mir sind wenige, gut lokalisierte Hinweise lieber, die meine Aufmerksamkeit wirklich " +
        "verdienen. Bitte sorgen Sie dafür, dass normale überlagernde Anatomie nicht zur Quelle vieler Fehlalarme wird: " +
        "Als letzte Sicherheitskontrolle soll es echte Auffälligkeiten von normaler Überlagerung unterscheiden, und wenn " +
        "es unsicher ist, wäre es mir lieber, es bleibt still, statt mich immer wieder zu normalen Regionen zurückzuschicken. " +
        "(Bei Bedarf dürfen Sie das per-round time budget von 30 min → 40 min erhöhen.)",
    },
  ],

  bannerSetupTitle: "Ihr Experiment wird eingerichtet.",
  bannerSetupBody:
    "Diese erste Phase kann manchmal bis zu 10 Minuten dauern, ohne dass sichtbar etwas passiert — das ist völlig normal, bitte haben Sie Geduld. " +
    "Wenn nach 20 Minuten immer noch keine Antwort erscheint, liegt möglicherweise ein Netzwerkproblem vor — bitte kontaktieren Sie " + CONTACT_EMAIL + ". " +
    "Möglicherweise bittet der Agent Sie noch, seinen Plan zu bestätigen, bevor er die iterative Optimierung startet — behalten Sie diese Seite " +
    "daher im Blick und übersehen Sie die Rückfrage nicht. " +
    "Sobald die Optimierung läuft, ist keinerlei weitere Eingabe von Ihnen nötig: " +
    "Sie können diese Seite bedenkenlos schließen und gehen. Wir benachrichtigen Sie per E-Mail, sobald Ihr Modell fertig ist.",
  bannerOptTitle: "Die iterative Optimierung läuft.",
  bannerOptBody:
    "Von Ihnen ist keine weitere Eingabe nötig. Sie können diese Seite bedenkenlos schließen und gehen — der Lauf wird auf dem Server " +
    "fortgesetzt, und wir benachrichtigen Sie per E-Mail, sobald Ihr Modell fertig ist.",

  transcriptHello:
    "Sagen Sie Hallo und machen Sie sich mit dem Assistenten vertraut. Wenn Sie bereit sind, senden Sie links Ihre Aufgabe ab — beachten Sie: Das ist nur einmal möglich.",
  agentBusy: "Agent arbeitet…",
  typePh: "Nachricht schreiben…",
  waitingPh: "Warte auf den Assistenten…",
  send: "Senden",
  chatEncourage:
    "Etwas unklar? Fragen Sie jederzeit alles — der Assistent hilft Ihnen gern beim Verständnis.",
  manualTitle: "Nutzungsanleitung",
  agentActivity: "Agent-Aktivität",
  stepCount: (n: number) => `${n} Schritt${n === 1 ? "" : "e"}`,

  runTraining: "Training",
  runPseudo: "Pseudo-GPU (Framework-Modus)",
  runRunning: "läuft…",
  runCompleted: "abgeschlossen",
  runBest: "bestes",
  runIteration: "Iteration",
  runEpoch: "Epoche",
};

const ja: Dict = {
  tagline: "臨床医主導のAI開発 — インタラクティブ・デモ",
  accessCode: "アクセスコード",
  accessCodePh: "アクセスコードを入力",
  email: "メールアドレス",
  emailPh: "モデル完成時にご連絡するメールアドレス",
  language: "表示言語",
  enter: "開始",
  badCode: "アクセスコードが正しくありません。",

  subtitle: "作りたいモデルを説明してください。自律エージェントが構築します。プログラミングは不要です。",
  endSession: "セッションを終了",
  completeBtn: "完了",
  stateActive: "実行中",
  stateCompleted: "完了",
  stateInterrupted: "中断",

  welcome:
    "✻ Clinical-Automata アシスタント（Claude Code CLI）へようこそ。\n\n" +
    "これはデモンストレーションです。「開始する」を押すと実際のセッションが\n" +
    "始まります。ご希望の臨床画像モデルを普段のことばで説明していただければ、\n" +
    "自律型コーディング・エージェントがそれを構築します。\n\n" +
    "プログラミングは不要です。開始するまで何も実行されません。",
  startUsing: "開始する",
  occupied: "現在このデモは別の方が使用中です。しばらくしてからもう一度お試しください。",

  howItWorks: "使い方",
  steps: [
    {
      title: "データセットを見る",
      body: "左側に利用可能なデータセットが表示されます。カードをクリックすると、どのようなアノテーションが含まれ、どのラベルカテゴリを対象としているかを確認できます。",
    },
    {
      title: "まず試してみる（任意）",
      body: "このようなアシスタントが初めてでしたら、まず中央のパネルで自由に会話してみてください。気になることは何でも質問できます。",
    },
    {
      title: "作りたいモデルを説明する",
      body: "準備ができたら、左の「タスクの説明」欄に作りたいものを入力して送信してください。普段の臨床のことばで構いません。コードを書く必要はありません。",
    },
    {
      title: "あとはお任せ — 離席して大丈夫です",
      body: "セットアップには時間がかかり、その後の最適化は自動で進みます。ページを閉じて問題ありません。モデルが完成したらメールでお知らせします。",
    },
  ],

  datasetsTitle: "利用可能なデータセット",
  dsDataset: "データセット",
  dsModality: "モダリティ",
  dsImages: "画像数",
  dsLicense: "ライセンス",
  dsAnnotations: "含まれるアノテーション",
  dsCategories: "ラベルカテゴリ",
  dsSource: "出典 ↗",

  taskLabel: "タスクの説明",
  taskPh:
    "作りたい臨床AIモデルについて、ご自身のことばで説明してください。" +
    "タスク、データ、臨床上の優先事項や懸念点などをお書きください。\n\n" +
    "書き方に迷ったら、下の「リクエスト例」をご覧ください。",
  taskSubmit: "タスクを送信して実験を開始",
  taskSetupWarn:
    "ご注意：送信後の「セットアップ」段階では、返答が返ってくるまでに最長で10分ほどかかることがあります。これは正常な動作です。" +
    "20分たっても応答がない場合はネットワークの問題の可能性がありますので、" + CONTACT_EMAIL + " までご連絡ください。",
  taskSubmittedLabel: "送信されたタスク",
  taskSubmittedFallback: "（送信済み）",
  taskSending: "送信中…",
  taskSendFailed: "送信できませんでした — 接続が切断されました。もう一度お試しください。",
  reconnecting: "再接続しています…",

  examplesTitle: "リクエスト例",
  copyRequest: "リクエストをコピー",
  copied: "コピーしました ✓",
  exampleModalHint:
    "これはあくまで例です。上のタスク欄に貼り付けて編集するか、ご自身のことばで書いてください。",
  examples: [
    {
      title: "気胸 — ショートカット学習を避ける",
      hint: "胸部X線 · 分類 · 交絡因子",
      text:
        "胸部X線写真における気胸の分類モデルを開発したいと考えています。" +
        "大きな懸念は、胸腔ドレーンの存在が強い交絡因子になりうる点です。" +
        "素朴なモデルは、気胸そのものの画像所見を認識するのではなく、単に胸腔ドレーンを検出することで" +
        "見かけ上の高い性能を達成してしまう可能性があります。このショートカット学習の問題を重視し、" +
        "胸腔ドレーンへの依存を減らし、気胸本来の特徴の学習を促す適切な方策を用いてください。",
    },
    {
      title: "肺葉と肺野 — 教育用",
      hint: "胸部X線 · セグメンテーション · 教育",
      text:
        "胸部X線の教育用に、肺葉と肺野のセグメンテーションモデルを作っていただけますか。" +
        "肺野（zone）と解剖学的な肺葉（lobe）を切り替えて表示できると、両者が同じ概念ではないことを" +
        "学生が理解しやすくなります。正面像では肺葉が互いに重なるため、すべての境界を確定的に示すのではなく、" +
        "出力に確信度（不確実性）の程度も示してほしいです。透明度を調整できる個別のオーバーレイがあれば理想的です。",
    },
    {
      title: "皮膚病変 — メラノーマを見逃さない",
      hint: "ダーモスコピー · 分類 · 感度",
      text:
        "ダーモスコピー画像から皮膚病変の診断を支援するAIモデルが欲しいです。" +
        "メラノーマを最優先とし、多少の特異度を犠牲にしてでも、できる限り見逃さないようにしてください。" +
        "メラノーマの感度は明示的に報告してください。",
    },
    {
      title: "胸部X線のセカンドリーダー — 見落としやすい領域",
      hint: "胸部X線 · 検出 · セーフティネット",
      text:
        "正面像の胸部X線写真では、肺尖部、心陰影後部、肋横隔膜角、横隔膜近傍といった読影しにくい領域に異常が" +
        "隠れていることがあり、正常な解剖構造が重なるため見落とされやすいです。自分で一次読影を終えた後に、" +
        "これらの見落としやすい領域だけをもう一度確認してくれる、第二読影者のようなツールが欲しいです。" +
        "画像全体の鑑別診断を一から並べ直す必要はありません。役立つのは、これらの領域に本当に見直す価値のある" +
        "所見があるときに、特に肋骨・鎖骨・心陰影・横隔膜に隠れている可能性のある所見について知らせてくれることです。" +
        "通知は短く具体的に、例えば「右肺尖をもう一度確認」「左肋横隔膜角を再確認」のようにしてほしく、" +
        "長いAI生成の説明は要りません。位置が明確で本当に見る価値のある少数の通知の方がありがたいです。" +
        "また、正常な重なりが多数の偽陽性の原因にならないようにしてください。最後の安全確認として、本当の異常と" +
        "正常な重なりを区別できるべきで、確信がなければ、正常な領域へ何度も戻させるより静かにしていてくれる方がよいです。" +
        "（必要であれば、per-round time budget を 30 min → 40 min に引き上げてかまいません。）",
    },
  ],

  bannerSetupTitle: "実験のセットアップ中です。",
  bannerSetupBody:
    "この最初の段階は、目に見える応答がほとんどないまま最長で10分ほどかかることがあります。これはまったく正常ですので、どうかお待ちください。" +
    "なお、エージェントが反復最適化を開始する前に、方針の確認を求めてくることがあります。この画面から目を離さず、質問を見逃さないようご注意ください。" +
    "最適化が始まれば、以降あなたの操作は一切不要です。このページを閉じて離席していただいて構いません。" +
    "モデルが完成しましたらメールでお知らせします。" +
    "20分たっても応答がない場合はネットワークの問題の可能性がありますので、" + CONTACT_EMAIL + " までご連絡ください。",
  bannerOptTitle: "反復最適化を実行中です。",
  bannerOptBody:
    "以降、あなたの操作は一切不要です。このページを閉じて離席していただいて構いません。処理はサーバー上で続行され、" +
    "モデルが完成しましたらメールでお知らせします。",

  transcriptHello:
    "まずは挨拶して、アシスタントの雰囲気をつかんでみてください。準備ができたら左側からタスクを送信してください（送信できるのは一度だけです）。",
  agentBusy: "エージェントが作業中です…",
  typePh: "メッセージを入力…",
  waitingPh: "アシスタントの応答を待っています…",
  send: "送信",
  chatEncourage:
    "わからないことがありますか？ いつでも何でも質問してください。アシスタントが理解を助けます。",
  manualTitle: "使用ガイド",
  agentActivity: "エージェントの作業",
  stepCount: (n: number) => `${n} ステップ`,

  runTraining: "学習中",
  runPseudo: "疑似GPU（フレームワーク・モード）",
  runRunning: "実行中…",
  runCompleted: "完了",
  runBest: "最良",
  runIteration: "反復",
  runEpoch: "エポック",
};

const zh: Dict = {
  tagline: "临床医生主导的 AI 开发 — 交互式演示",
  accessCode: "访问码",
  accessCodePh: "请输入您的访问码",
  email: "邮箱",
  emailPh: "模型完成后我们将发送邮件到这个地址",
  language: "界面语言",
  enter: "进入",
  badCode: "访问码不正确。",

  subtitle: "描述您想要的模型，自主智能体会替您构建。无需编程。",
  endSession: "结束会话",
  completeBtn: "完成",
  stateActive: "进行中",
  stateCompleted: "已完成",
  stateInterrupted: "已中断",

  welcome:
    "✻ 欢迎使用 Clinical-Automata 助手（Claude Code CLI）。\n\n" +
    "这是一个演示。点击「开始使用」即可开启一次真实会话。\n" +
    "您可以用日常语言描述想要的临床影像模型 —— 一个自主编码智能体\n" +
    "会替您把它构建出来。\n\n" +
    "无需任何编程。在您开始之前，不会运行任何程序。",
  startUsing: "开始使用",
  occupied: "该演示当前正被另一位访客使用，请稍后再试。",

  howItWorks: "使用步骤",
  steps: [
    {
      title: "浏览数据集",
      body: "左侧是可用的数据集。点击任意卡片，可以查看它提供了哪些标注、覆盖了哪些标签类别。",
    },
    {
      title: "先聊几句（可选）",
      body: "第一次用这类助手？可以先在中间的面板里自由对话 —— 任何好奇的问题都可以问。",
    },
    {
      title: "描述您想要的模型",
      body: "准备好之后，在左侧「任务描述」框里写下您想构建的东西并提交。用平常的临床语言就行 —— 不需要写代码。",
    },
    {
      title: "交给它 —— 您可以离开",
      body: "初始设置需要一些时间，之后的优化会自动进行。您可以放心关掉网页 —— 模型完成后我们会发邮件通知您。",
    },
  ],

  datasetsTitle: "可用数据集",
  dsDataset: "数据集",
  dsModality: "模态",
  dsImages: "图像数量",
  dsLicense: "许可协议",
  dsAnnotations: "提供的标注",
  dsCategories: "标签类别",
  dsSource: "来源 ↗",

  taskLabel: "任务描述",
  taskPh:
    "请用您自己的话描述想要构建的临床 AI 模型 —— 任务是什么、用什么数据、" +
    "以及任何临床上的优先事项或顾虑。\n\n" +
    "不确定怎么写？可以看看下面的示例请求。",
  taskSubmit: "提交任务并开始实验",
  taskSetupWarn:
    "请注意：提交后的「初始设置」阶段有时会长达 10 分钟才会有回应，这是正常现象。" +
    "如果 20 分钟仍然没有回复，说明可能存在网络问题，请联系 " + CONTACT_EMAIL + "。",
  taskSubmittedLabel: "您已提交的任务",
  taskSubmittedFallback: "（已提交）",
  taskSending: "正在提交…",
  taskSendFailed: "发送失败 —— 连接已断开，请重试。",
  reconnecting: "正在重新连接…",

  examplesTitle: "示例请求",
  copyRequest: "复制请求",
  copied: "已复制 ✓",
  exampleModalHint: "这只是一个示例 —— 您可以把它粘贴到上方的任务框里修改，或者自己重新写。",
  examples: [
    {
      title: "气胸 —— 避免捷径学习",
      hint: "胸部 X 线 · 分类 · 混杂因素",
      text:
        "我想开发一个用于胸片气胸检测的分类模型。一个主要顾虑是：胸腔引流管的存在可能构成很强的混杂因素 —— " +
        "简单的模型可能仅仅通过识别引流管就取得看似不错的性能，而并没有真正学到气胸本身的影像学特征。" +
        "请认真对待这个捷径学习问题，采用适当的策略来降低模型对引流管的依赖，促使它学习气胸真正的特征。",
    },
    {
      title: "肺叶与肺野 —— 用于教学",
      hint: "胸部 X 线 · 分割 · 教学",
      text:
        "能否提供一个用于胸片教学的肺叶与肺野分割模型？如果这个工具能够在肺野（zone）和解剖学肺叶（lobe）之间切换，" +
        "会很有帮助 —— 这样学生能理解两者并不是同一个概念。由于在正位胸片上肺叶彼此重叠，模型还应当在输出上给出" +
        "置信度或不确定性的程度，而不是把每一条边界都呈现为确定的。若能提供可调透明度的独立叠加层，对教学是最理想的。",
    },
    {
      title: "皮肤病变 —— 不要漏诊黑色素瘤",
      hint: "皮肤镜 · 分类 · 灵敏度",
      text:
        "我想要一个能根据皮肤镜图像辅助诊断皮肤病变的 AI 模型。请把黑色素瘤放在首位：" +
        "尽最大努力不要漏诊，即使为此牺牲一些特异度也可以。请明确报告黑色素瘤的灵敏度。",
    },
    {
      title: "胸片第二读片者 —— 容易漏看的区域",
      hint: "胸部 X 线 · 检测 · 安全网",
      text:
        "在正位胸片上，有些异常容易藏在比较难看的区域里 —— 肺尖、心后区、肋膈角以及靠近膈肌的位置 —— " +
        "因为正常解剖结构重叠，很容易被忽略。我希望有一个工具，在我完成自己的初步阅片之后作为第二读片者，" +
        "专门再检查这些容易漏看的区域。我不需要它把整张片子重新列一遍鉴别诊断；对我更有帮助的是，当这些区域里" +
        "确实有值得再看一眼的东西时提醒我，尤其是那些可能被肋骨、锁骨、心影或膈肌遮住的异常。提醒要尽量简短而具体，" +
        "比如“再看一下右肺尖”或“重新检查左侧肋膈角”，而不是一大段 AI 生成的解释。我宁愿收到少量、位置明确、" +
        "确实值得我重新看的提醒。同时，请不要让正常解剖结构本身成为大量误报的来源：作为最后一道安全检查，" +
        "它应该能区分真正的异常和正常结构重叠；如果它自己都不确定，我宁愿它保持安静，也不要不断让我重新检查正常区域。" +
        "（如有需要，可以把 per-round time budget 从 30 min → 40 min。）",
    },
  ],

  bannerSetupTitle: "正在为您的实验做初始设置。",
  bannerSetupBody:
    "这个最初的阶段有时会长达 10 分钟，期间几乎看不到任何回应 —— 这完全正常，请耐心等待。" +
    "如果 20 分钟仍然没有回复，说明可能存在网络问题，请联系 " + CONTACT_EMAIL + "。" +
    "另外，智能体有时会在正式开始迭代优化之前，请您确认它的方案，所以请留意本页面，别错过它的提问。" +
    "一旦进入迭代优化阶段，就完全不再需要您的任何介入：您可以放心关掉本页面离开。" +
    "模型完成后，我们会发邮件通知您。",
  bannerOptTitle: "迭代优化正在进行中。",
  bannerOptBody:
    "接下来完全不需要您的介入。您可以放心关掉本页面离开 —— 任务会在服务器上继续运行，" +
    "模型完成后我们会发邮件通知您。",

  transcriptHello: "先打个招呼、熟悉一下这个助手。准备好之后，从左侧提交您的任务 —— 注意任务只能提交一次。",
  agentBusy: "智能体正在工作…",
  typePh: "输入消息…",
  waitingPh: "正在等待助手回应…",
  send: "发送",
  chatEncourage: "有什么不清楚的吗？随时都可以提问 —— 助手随时在这里帮您理解。",
  manualTitle: "使用指南",
  agentActivity: "智能体活动",
  stepCount: (n: number) => `${n} 步`,

  runTraining: "训练中",
  runPseudo: "伪 GPU（框架模式）",
  runRunning: "运行中…",
  runCompleted: "已完成",
  runBest: "最佳",
  runIteration: "迭代",
  runEpoch: "轮次",
};

export const STRINGS: Record<Lang, Dict> = { en, de, ja, zh };

export const LangContext = createContext<Dict>(en);
export const useT = (): Dict => useContext(LangContext);
