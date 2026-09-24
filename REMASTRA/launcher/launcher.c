/*
 * REMASTRA — lanceur Windows (Remastra.exe)
 *
 *  - Premier lancement : choix de la langue (English / Français), puis
 *    installation automatique (install.bat) de l'environnement « runtime\ »
 *    et des moteurs IA, dans la langue choisie ; l'application démarre
 *    ensuite directement dans cette langue (--lang xx).
 *  - Lancements suivants : démarre l'application sans console
 *    (runtime\Scripts\pythonw.exe -m remastra), qui affiche d'abord la
 *    fenêtre de choix de la langue puis l'interface.
 *  - Les fichiers glissés sur l'icône sont transmis à l'application.
 *
 *  Compilation : build_launcher.sh (Linux, MinGW-w64) ou build_launcher.bat (Windows, MinGW).
 */
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#include <windows.h>
#include <commctrl.h>
#include <shellapi.h>
#include <wchar.h>

#define APP_TITLE L"REMASTRA — AI Audio Remaster Studio"
#define ID_EN 1001
#define ID_FR 1002

static wchar_t g_dir[MAX_PATH];
static int g_en = 0;   /* 1 = anglais */

#define S(fr, en) (g_en ? (en) : (fr))

static BOOL file_exists(const wchar_t *p) {
    DWORD a = GetFileAttributesW(p);
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

static void join(wchar_t *out, size_t n, const wchar_t *rel) {
    _snwprintf(out, n, L"%s\\%s", g_dir, rel);
    out[n - 1] = 0;
}

static DWORD run(const wchar_t *cmdline, BOOL wait, BOOL console) {
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    wchar_t buf[8192];
    DWORD code = 0;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);
    wcsncpy(buf, cmdline, 8191);
    buf[8191] = 0;
    if (!CreateProcessW(NULL, buf, NULL, NULL, FALSE,
                        console ? CREATE_NEW_CONSOLE : CREATE_NO_WINDOW,
                        NULL, g_dir, &si, &pi)) {
        return (DWORD)-1;
    }
    if (wait) {
        WaitForSingleObject(pi.hProcess, INFINITE);
        GetExitCodeProcess(pi.hProcess, &code);
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return code;
}

/* Fenêtre native de choix de la langue (TaskDialog, boutons « command link »).
 * Retourne 1 = anglais, 0 = français, -1 = annulé. */
typedef HRESULT (WINAPI *PFN_TDI)(const TASKDIALOGCONFIG *, int *, int *, BOOL *);

static int ask_language(void) {
    TASKDIALOG_BUTTON buttons[2] = {
        {ID_EN, L"English\nContinue in English"},
        {ID_FR, L"Français\nContinuer en français"},
    };
    TASKDIALOGCONFIG cfg;
    int pressed = 0;
    HMODULE comctl = LoadLibraryW(L"comctl32.dll");
    PFN_TDI tdi = comctl ? (PFN_TDI)GetProcAddress(comctl, "TaskDialogIndirect") : NULL;
    LANGID ui = GetUserDefaultUILanguage();
    int default_en = PRIMARYLANGID(ui) != LANG_FRENCH;

    if (!tdi) {  /* repli : boîte de message Oui = English / Non = Français */
        int r = MessageBoxW(NULL, L"English?  (Yes = English / Non = Français)",
                            APP_TITLE, MB_YESNOCANCEL | MB_ICONQUESTION);
        return r == IDYES ? 1 : r == IDNO ? 0 : -1;
    }
    ZeroMemory(&cfg, sizeof(cfg));
    cfg.cbSize = sizeof(cfg);
    cfg.hInstance = GetModuleHandleW(NULL);
    cfg.dwFlags = TDF_USE_COMMAND_LINKS | TDF_ALLOW_DIALOG_CANCELLATION | TDF_POSITION_RELATIVE_TO_WINDOW;
    cfg.pszWindowTitle = APP_TITLE;
    cfg.pszMainIcon = MAKEINTRESOURCEW(1);
    cfg.dwFlags |= TDF_USE_HICON_MAIN;
    cfg.hMainIcon = LoadIconW(GetModuleHandleW(NULL), MAKEINTRESOURCEW(1));
    cfg.pszMainInstruction = L"Choose your language  ·  Choisissez votre langue";
    cfg.pszContent = L"REMASTRA — AI Audio Remaster Studio";
    cfg.pButtons = buttons;
    cfg.cButtons = 2;
    cfg.nDefaultButton = default_en ? ID_EN : ID_FR;
    cfg.dwCommonButtons = TDCBF_CANCEL_BUTTON;
    if (FAILED(tdi(&cfg, &pressed, NULL, NULL))) return default_en;
    if (pressed == ID_EN) return 1;
    if (pressed == ID_FR) return 0;
    return -1;
}

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE hPrev, PWSTR pCmd, int nShow) {
    wchar_t pyw[MAX_PATH], ready[MAX_PATH], inst[MAX_PATH], pkg[MAX_PATH];
    wchar_t cmd[8192];
    int argc = 0, i, fresh_install = 0;
    LPWSTR *argv;
    (void)hInst; (void)hPrev; (void)pCmd; (void)nShow;

    GetModuleFileNameW(NULL, g_dir, MAX_PATH);
    wchar_t *slash = wcsrchr(g_dir, L'\\');
    if (slash) *slash = 0;
    g_en = PRIMARYLANGID(GetUserDefaultUILanguage()) != LANG_FRENCH;

    join(pyw, MAX_PATH, L"runtime\\Scripts\\pythonw.exe");
    join(ready, MAX_PATH, L"runtime\\.ready");
    join(inst, MAX_PATH, L"install.bat");
    join(pkg, MAX_PATH, L"remastra\\__main__.py");

    if (!file_exists(pkg)) {
        MessageBoxW(NULL,
            S(L"Le dossier « remastra » est introuvable à côté de Remastra.exe.\n\n"
              L"Gardez Remastra.exe dans le dossier REMASTRA d'origine.",
              L"The \"remastra\" folder was not found next to Remastra.exe.\n\n"
              L"Keep Remastra.exe inside the original REMASTRA folder."),
            APP_TITLE, MB_ICONERROR);
        return 1;
    }

    if (!file_exists(pyw) || !file_exists(ready)) {
        /* Premier lancement : la langue est choisie ici, avant l'installation */
        int lang = ask_language();
        if (lang < 0) return 0;
        g_en = lang;
        int r = MessageBoxW(NULL,
            S(L"Bienvenue dans REMASTRA !\n\n"
              L"Premier lancement : les composants doivent être installés\n"
              L"(Python, interface, moteurs IA Demucs, BS-RoFormer).\n\n"
              L"• Connexion Internet requise\n"
              L"• Espace disque : environ 3 à 6 Go\n"
              L"• Durée : 5 à 20 minutes selon la connexion\n\n"
              L"Lancer l'installation maintenant ?",
              L"Welcome to REMASTRA!\n\n"
              L"First launch: the components need to be installed\n"
              L"(Python, user interface, Demucs and BS-RoFormer AI engines).\n\n"
              L"• Internet connection required\n"
              L"• Disk space: about 3 to 6 GB\n"
              L"• Time: 5 to 20 minutes depending on your connection\n\n"
              L"Start the installation now?"),
            APP_TITLE, MB_ICONINFORMATION | MB_YESNO);
        if (r != IDYES) return 0;
        if (!file_exists(inst)) {
            MessageBoxW(NULL, S(L"install.bat est introuvable.", L"install.bat was not found."),
                        APP_TITLE, MB_ICONERROR);
            return 1;
        }
        _snwprintf(cmd, 8192, L"cmd.exe /c \"\"%s\" /launcher /lang:%s\"", inst, g_en ? L"en" : L"fr");
        run(cmd, TRUE, TRUE);
        if (!file_exists(pyw) || !file_exists(ready)) {
            MessageBoxW(NULL,
                S(L"L'installation n'a pas pu se terminer.\n\n"
                  L"Consultez le fichier install_log.txt dans le dossier REMASTRA,\n"
                  L"puis relancez Remastra.exe.",
                  L"The installation could not be completed.\n\n"
                  L"Check install_log.txt in the REMASTRA folder,\n"
                  L"then run Remastra.exe again."),
                APP_TITLE, MB_ICONERROR);
            return 1;
        }
        fresh_install = 1;
    }

    /* Démarre l'application : pythonw -m remastra [--lang xx] [fichiers…]
     * Sans --lang, l'application affiche sa fenêtre de choix de la langue. */
    if (fresh_install)
        _snwprintf(cmd, 8192, L"\"%s\" -m remastra --lang %s", pyw, g_en ? L"en" : L"fr");
    else
        _snwprintf(cmd, 8192, L"\"%s\" -m remastra", pyw);
    argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    for (i = 1; argv && i < argc; ++i) {
        size_t len = wcslen(cmd);
        _snwprintf(cmd + len, 8192 - len, L" \"%s\"", argv[i]);
    }
    if (argv) LocalFree(argv);
    cmd[8191] = 0;
    if (run(cmd, FALSE, FALSE) == (DWORD)-1) {
        MessageBoxW(NULL,
            S(L"Impossible de démarrer REMASTRA (runtime corrompu ?).\n"
              L"Supprimez le dossier « runtime » puis relancez Remastra.exe.",
              L"Could not start REMASTRA (corrupted runtime?).\n"
              L"Delete the \"runtime\" folder, then run Remastra.exe again."),
            APP_TITLE, MB_ICONERROR);
        return 1;
    }
    return 0;
}
