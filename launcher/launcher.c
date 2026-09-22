/*
 * REMASTRA — lanceur Windows (Remastra.exe)
 *
 *  - Premier lancement : propose l'installation automatique (install.bat)
 *    qui crée l'environnement « runtime\ » et installe les moteurs IA.
 *  - Lancements suivants : démarre l'application sans console
 *    (runtime\Scripts\pythonw.exe -m remastra).
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
#include <windows.h>
#include <shellapi.h>
#include <wchar.h>

#define APP_TITLE L"REMASTRA — AI Audio Remaster Studio"

static wchar_t g_dir[MAX_PATH];

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

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE hPrev, PWSTR pCmd, int nShow) {
    wchar_t pyw[MAX_PATH], ready[MAX_PATH], inst[MAX_PATH], pkg[MAX_PATH];
    wchar_t cmd[8192];
    int argc = 0, i;
    LPWSTR *argv;
    (void)hInst; (void)hPrev; (void)pCmd; (void)nShow;

    GetModuleFileNameW(NULL, g_dir, MAX_PATH);
    wchar_t *slash = wcsrchr(g_dir, L'\\');
    if (slash) *slash = 0;

    join(pyw, MAX_PATH, L"runtime\\Scripts\\pythonw.exe");
    join(ready, MAX_PATH, L"runtime\\.ready");
    join(inst, MAX_PATH, L"install.bat");
    join(pkg, MAX_PATH, L"remastra\\__main__.py");

    if (!file_exists(pkg)) {
        MessageBoxW(NULL,
            L"Le dossier « remastra » est introuvable à côté de Remastra.exe.\n\n"
            L"Gardez Remastra.exe dans le dossier REMASTRA d'origine.",
            APP_TITLE, MB_ICONERROR);
        return 1;
    }

    if (!file_exists(pyw) || !file_exists(ready)) {
        int r = MessageBoxW(NULL,
            L"Bienvenue dans REMASTRA !\n\n"
            L"Premier lancement : les composants doivent être installés\n"
            L"(Python, interface, moteurs IA Demucs & DeepFilterNet).\n\n"
            L"• Connexion Internet requise\n"
            L"• Espace disque : environ 3 à 6 Go\n"
            L"• Durée : 5 à 20 minutes selon la connexion\n\n"
            L"Lancer l'installation maintenant ?",
            APP_TITLE, MB_ICONINFORMATION | MB_YESNO);
        if (r != IDYES) return 0;
        if (!file_exists(inst)) {
            MessageBoxW(NULL, L"install.bat est introuvable.", APP_TITLE, MB_ICONERROR);
            return 1;
        }
        _snwprintf(cmd, 8192, L"cmd.exe /c \"\"%s\" /launcher\"", inst);
        run(cmd, TRUE, TRUE);
        if (!file_exists(pyw) || !file_exists(ready)) {
            MessageBoxW(NULL,
                L"L'installation n'a pas pu se terminer.\n\n"
                L"Consultez le fichier install_log.txt dans le dossier REMASTRA,\n"
                L"puis relancez Remastra.exe.",
                APP_TITLE, MB_ICONERROR);
            return 1;
        }
    }

    /* Démarre l'application : pythonw -m remastra [fichiers…] */
    _snwprintf(cmd, 8192, L"\"%s\" -m remastra", pyw);
    argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    for (i = 1; argv && i < argc; ++i) {
        size_t len = wcslen(cmd);
        _snwprintf(cmd + len, 8192 - len, L" \"%s\"", argv[i]);
    }
    if (argv) LocalFree(argv);
    cmd[8191] = 0;
    if (run(cmd, FALSE, FALSE) == (DWORD)-1) {
        MessageBoxW(NULL, L"Impossible de démarrer REMASTRA (runtime corrompu ?).\n"
                          L"Supprimez le dossier « runtime » puis relancez Remastra.exe.",
                    APP_TITLE, MB_ICONERROR);
        return 1;
    }
    return 0;
}
