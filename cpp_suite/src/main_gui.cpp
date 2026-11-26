#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <dwmapi.h>
#include <d3d11.h>
#include <tchar.h>
#include <algorithm>

#include "../lib/imgui/imgui.h"
#include "../lib/imgui/imgui_impl_win32.h"
#include "../lib/imgui/imgui_impl_dx11.h"

#include "app_gui.h"
#include "constants.h"
#include "ui_observatory.h"

#ifndef DWMWA_USE_IMMERSIVE_DARK_MODE
#define DWMWA_USE_IMMERSIVE_DARK_MODE 20
#endif
#ifndef DWMWA_WINDOW_CORNER_PREFERENCE
#define DWMWA_WINDOW_CORNER_PREFERENCE 33
#endif
#ifndef DWMWA_BORDER_COLOR
#define DWMWA_BORDER_COLOR 34
#endif
#ifndef DWMWA_CAPTION_COLOR
#define DWMWA_CAPTION_COLOR 35
#endif
#ifndef DWMWA_TEXT_COLOR
#define DWMWA_TEXT_COLOR 36
#endif

namespace {
void apply_win32_dark_chrome(HWND hwnd) {
    if (!hwnd) return;
    const BOOL dark_mode = TRUE;
    if (FAILED(DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &dark_mode, sizeof(dark_mode)))) {
        constexpr DWORD kDwmwaDarkModeLegacy19 = 19;
        DwmSetWindowAttribute(hwnd, kDwmwaDarkModeLegacy19, &dark_mode, sizeof(dark_mode));
    }
    const DWORD corner_pref = 2; // DWMWCP_ROUND on Windows 11
    const COLORREF caption  = RGB(15, 20, 27);   // #0F141B HeaderBg
    const COLORREF border   = RGB(35, 45, 59);   // Subtle Slate border
    const COLORREF text_col = RGB(230, 237, 243);
    DwmSetWindowAttribute(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, &corner_pref, sizeof(corner_pref));
    DwmSetWindowAttribute(hwnd, DWMWA_CAPTION_COLOR, &caption, sizeof(caption));
    DwmSetWindowAttribute(hwnd, DWMWA_BORDER_COLOR, &border, sizeof(border));
    DwmSetWindowAttribute(hwnd, DWMWA_TEXT_COLOR, &text_col, sizeof(text_col));
}
} // namespace

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

static ID3D11Device* g_pd3dDevice = nullptr;
static ID3D11DeviceContext* g_pd3dDeviceContext = nullptr;
static IDXGISwapChain* g_pSwapChain = nullptr;
static ID3D11RenderTargetView* g_mainRenderTargetView = nullptr;
static UINT g_ResizeWidth = 0, g_ResizeHeight = 0;
static float g_PendingDpiScale = 0.0f;

static bool CreateRenderTarget() {
    if (!g_pSwapChain || !g_pd3dDevice) return false;
    ID3D11Texture2D* pBackBuffer = nullptr;
    HRESULT hr = g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    if (FAILED(hr) || !pBackBuffer) {
        return false;
    }
    hr = g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
    pBackBuffer->Release();
    if (FAILED(hr)) {
        g_mainRenderTargetView = nullptr;
        return false;
    }
    return true;
}

static void CleanupRenderTarget() {
    if (g_mainRenderTargetView) {
        g_mainRenderTargetView->Release();
        g_mainRenderTargetView = nullptr;
    }
}

static bool CreateDeviceD3D(HWND hWnd) {
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT createDeviceFlags = 0;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_0,
    };

    HRESULT res = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags,
        featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain,
        &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext
    );
    if (res == DXGI_ERROR_UNSUPPORTED) {
        res = D3D11CreateDeviceAndSwapChain(
            nullptr, D3D_DRIVER_TYPE_WARP, nullptr, createDeviceFlags,
            featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain,
            &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext
        );
    }
    if (FAILED(res)) return false;

    return CreateRenderTarget();
}

static void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = nullptr; }
}

static LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg) {
        case WM_SIZE:
            if (wParam == SIZE_MINIMIZED)
                return 0;
            g_ResizeWidth = (UINT)LOWORD(lParam);
            g_ResizeHeight = (UINT)HIWORD(lParam);
            return 0;
        case WM_DPICHANGED: {
            // Defer ImGui style mutation until outside any active ImGui frame
            // (prevents style corruption if WM_DPICHANGED fires during synchronous GetOpenFileNameW).
            const float new_dpi = static_cast<float>(LOWORD(wParam)) / 96.0f;
            if (new_dpi >= 0.5f && new_dpi <= 8.0f) {
                g_PendingDpiScale = new_dpi;
            }
            RECT* const suggested = reinterpret_cast<RECT*>(lParam);
            if (suggested) {
                SetWindowPos(hWnd, nullptr,
                             suggested->left, suggested->top,
                             suggested->right - suggested->left,
                             suggested->bottom - suggested->top,
                             SWP_NOZORDER | SWP_NOACTIVATE);
            }
            return 0;
        }
        case WM_SYSCOMMAND:
            if ((wParam & 0xfff0) == SC_KEYMENU)
                return 0;
            break;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hWnd, msg, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    ImGui_ImplWin32_EnableDpiAwareness();

    HBRUSH bg_brush = CreateSolidBrush(RGB(13, 17, 23)); // #0D1117 prevents white flash on startup/resize
    HICON app_icon = LoadIconW(hInstance, MAKEINTRESOURCEW(101));
    WNDCLASSEXW wc = {
        sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L, hInstance, app_icon, nullptr, bg_brush, nullptr,
        L"HydrogenWindowClass", app_icon
    };
    RegisterClassExW(&wc);

    float init_dpi = ImGui_ImplWin32_GetDpiScaleForMonitor(
        ::MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY));
    if (init_dpi < 1.0f) init_dpi = 1.0f;

    RECT work_area{0, 0, 1920, 1080};
    SystemParametersInfoW(SPI_GETWORKAREA, 0, &work_area, 0);
    int max_w = std::max(800, static_cast<int>((work_area.right - work_area.left) - 40));
    int max_h = std::max(600, static_cast<int>((work_area.bottom - work_area.top) - 40));
    int win_w = std::min(max_w, static_cast<int>(1440.0f * init_dpi));
    int win_h = std::min(max_h, static_cast<int>(920.0f * init_dpi));
    int win_x = static_cast<int>(work_area.left) + std::max(0, (max_w + 40 - win_w) / 2);
    int win_y = static_cast<int>(work_area.top) + std::max(0, (max_h + 40 - win_h) / 2);

    HWND hwnd = CreateWindowW(
        wc.lpszClassName,
        L"Hydrogen - 21cm H I SDR & Analysis",
        WS_OVERLAPPEDWINDOW, win_x, win_y, win_w, win_h, nullptr, nullptr, wc.hInstance, nullptr
    );
    apply_win32_dark_chrome(hwnd);

    if (!CreateDeviceD3D(hwnd)) {
        CleanupDeviceD3D();
        DestroyWindow(hwnd);
        UnregisterClassW(wc.lpszClassName, wc.hInstance);
        if (bg_brush) DeleteObject(bg_brush);
        return 1;
    }

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigDpiScaleFonts = true;
    io.IniFilename = nullptr;

    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);
    ImGui::GetStyle().FontScaleDpi = ImGui_ImplWin32_GetDpiScaleForHwnd(hwnd);

    AppGui gui;
    gui.set_owner_hwnd(hwnd);
    gui.init_d3d_resources(g_pd3dDevice);

    auto recover_d3d_device = [&]() -> bool {
        gui.release_d3d_resources();
        ImGui_ImplDX11_Shutdown();
        CleanupDeviceD3D();
        if (!CreateDeviceD3D(hwnd)) {
            return false;
        }
        ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);
        gui.init_d3d_resources(g_pd3dDevice);
        return true;
    };

    bool done = false;
    bool occluded = false;
    bool device_lost = false;
    while (!done) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
            if (msg.message == WM_QUIT)
                done = true;
        }
        if (done) break;

        if (g_PendingDpiScale > 0.0f) {
            ImGui::GetStyle().FontScaleDpi = g_PendingDpiScale;
            ui_obs::apply_observatory_slate_style(g_PendingDpiScale);
            g_PendingDpiScale = 0.0f;
        }

        if (device_lost) {
            if (!recover_d3d_device()) {
                Sleep(100);
                continue;
            }
            device_lost = false;
        }

        if (IsIconic(hwnd)) {
            Sleep(50);
            continue;
        }
        if (occluded) {
            HRESULT hr_test = g_pSwapChain->Present(0, DXGI_PRESENT_TEST);
            if (hr_test == DXGI_STATUS_OCCLUDED) {
                Sleep(50);
                continue;
            }
            if (hr_test == DXGI_ERROR_DEVICE_REMOVED || hr_test == DXGI_ERROR_DEVICE_RESET) {
                device_lost = true;
                continue;
            }
            occluded = false;
        }

        if (g_ResizeWidth != 0 && g_ResizeHeight != 0) {
            CleanupRenderTarget();
            HRESULT hr_resize = g_pSwapChain->ResizeBuffers(0, g_ResizeWidth, g_ResizeHeight, DXGI_FORMAT_UNKNOWN, 0);
            g_ResizeWidth = g_ResizeHeight = 0;
            if (hr_resize == DXGI_ERROR_DEVICE_REMOVED || hr_resize == DXGI_ERROR_DEVICE_RESET) {
                device_lost = true;
                continue;
            }
            if (SUCCEEDED(hr_resize)) {
                CreateRenderTarget();
            }
        }

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        RECT rect;
        GetClientRect(hwnd, &rect);
        float width = static_cast<float>(rect.right - rect.left);
        float height = static_cast<float>(rect.bottom - rect.top);

        gui.render_frame(g_pd3dDeviceContext, width, height);

        ImGui::Render();
        const float clear_color[4] = {
            astro::theme::BG_R, astro::theme::BG_G, astro::theme::BG_B, astro::theme::BG_A
        };
        if (g_mainRenderTargetView) {
            g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
            g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color);
            ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        }

        HRESULT hr_pres = g_pSwapChain->Present(1, 0);
        if (hr_pres == DXGI_STATUS_OCCLUDED) {
            occluded = true;
        } else if (hr_pres == DXGI_ERROR_DEVICE_REMOVED || hr_pres == DXGI_ERROR_DEVICE_RESET) {
            device_lost = true;
        }
    }

    gui.release_d3d_resources();

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    DestroyWindow(hwnd);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);
    if (bg_brush) DeleteObject(bg_brush);

    return 0;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    return WinMain(GetModuleHandle(nullptr), nullptr, GetCommandLineA(), SW_SHOWDEFAULT);
}
