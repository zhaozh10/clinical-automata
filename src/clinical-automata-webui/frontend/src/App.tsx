import { useState } from "react";
import Login from "./components/Login";
import Studio from "./components/Studio";
import { DEFAULT_LANG, LangContext, STRINGS, type Lang } from "./i18n";

export default function App() {
  // Require the access code on every visit: start unauthenticated
  // and only enter the studio after a successful code entry.
  const [authed, setAuthed] = useState(false);
  const [lang, setLang] = useState<Lang>(DEFAULT_LANG);

  // Ending a session hands the demo back to the next visitor: drop the
  // identity AND the language, so they land on a clean English login page.
  const endSession = () => {
    setAuthed(false);
    setLang(DEFAULT_LANG);
  };

  return (
    <LangContext.Provider value={STRINGS[lang]}>
      {authed ? (
        <Studio onEnded={endSession} />
      ) : (
        <Login lang={lang} onLang={setLang} onAuthed={() => setAuthed(true)} />
      )}
    </LangContext.Provider>
  );
}
