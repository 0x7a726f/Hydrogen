#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#ifndef _WIN32
#include <sys/wait.h>
#endif

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST_ASSERT(cond, msg) do { \
    g_tests_run++; \
    if (cond) { \
        g_tests_passed++; \
    } else { \
        std::cerr << "  [FAIL] " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
    } \
} while(0)

#define ASSERT_TRUE(a, msg)  TEST_ASSERT((a), msg)
#define ASSERT_FALSE(a, msg) TEST_ASSERT(!(a), msg)
#define ASSERT_EQ(a, b, msg) TEST_ASSERT((a) == (b), msg)

struct CommandResult {
    int exit_code;
    std::string output;
};

static CommandResult run_cli(const std::string& args) {
    std::string exe_path;
#ifdef _WIN32
    const std::vector<std::string> candidates = {
        "bin\\radio_suite_cli.exe",
        "radio_suite_cli.exe",
        "..\\bin\\radio_suite_cli.exe"
    };
#else
    const std::vector<std::string> candidates = {
        "bin/radio_suite_cli",
        "./radio_suite_cli",
        "../bin/radio_suite_cli",
        "build/bin/radio_suite_cli"
    };
#endif
    for (const auto& c : candidates) {
        if (std::filesystem::exists(c)) {
            exe_path = c;
            break;
        }
    }
    if (exe_path.empty()) {
        exe_path = candidates.front();
    }

#ifdef _WIN32
    std::string cmd = "\"\"" + exe_path + "\" " + args + " 2>&1\"";
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    std::string cmd = "\"" + exe_path + "\" " + args + " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) return {-1, "Failed to execute pipe"};

    char buffer[512];
    std::string out;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        out += buffer;
    }
#ifdef _WIN32
    int rc = _pclose(pipe);
#else
    int raw_rc = pclose(pipe);
    int rc = WIFEXITED(raw_rc) ? WEXITSTATUS(raw_rc) : -1;
#endif
    return {rc, out};
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  RUNNING TEST SUITE: test_cli_commands (CLI Subcommands & Flag Validation)\n";
    std::cout << "======================================================================\n";

    std::string test_dir = "tests/test_tmp_cli";
    std::string out_dir  = "tests/test_out_cli";
    std::error_code ec;
    std::filesystem::remove_all(test_dir, ec);
    std::filesystem::remove_all(out_dir, ec);
    std::filesystem::create_directories(test_dir, ec);
    std::filesystem::create_directories(out_dir, ec);

    // -----------------------------------------------------------------------
    // TEST 1: Help and Usage Banners
    // -----------------------------------------------------------------------
    std::cout << "\n[Test 1] Testing Help & Banner Flags...\n";
    {
        auto res = run_cli("--help");
        ASSERT_EQ(res.exit_code, 0, "CLI --help exits with code 0");
        ASSERT_TRUE(res.output.find("radio_suite_cli") != std::string::npos, "Help identifies radio_suite_cli");
        ASSERT_TRUE(res.output.find("Record Options:") != std::string::npos, "Help lists record subcommand");
        ASSERT_TRUE(res.output.find("Analyze Options:") != std::string::npos, "Help lists analyze subcommand");

        auto res2 = run_cli("-h");
        ASSERT_EQ(res2.exit_code, 0, "CLI -h exits with code 0");

        auto res_rec_h = run_cli("record --help");
        ASSERT_EQ(res_rec_h.exit_code, 0, "CLI record --help exits with code 0");

        auto res_ana_h = run_cli("analyze --help");
        ASSERT_EQ(res_ana_h.exit_code, 0, "CLI analyze --help exits with code 0");
    }

    // -----------------------------------------------------------------------
    // TEST 2: Record Subcommand Argument Validation
    // -----------------------------------------------------------------------
    std::cout << "\n[Test 2] Testing Record Subcommand Flag Validation...\n";
    {
        auto res_bad_flag = run_cli("record --nonexistent-option");
        ASSERT_EQ(res_bad_flag.exit_code, 1, "Unrecognized option exits with error code 1");
        ASSERT_TRUE(res_bad_flag.output.find("Unrecognized option for record") != std::string::npos,
                    "Reports unrecognized option error message");

        auto res_m_fc = run_cli("record --fc");
        ASSERT_EQ(res_m_fc.exit_code, 1, "Missing --fc argument exits with error code 1");
        ASSERT_TRUE(res_m_fc.output.find("Missing value for --fc") != std::string::npos, "Reports missing value error");

        auto res_m_fs = run_cli("record --fs");
        ASSERT_EQ(res_m_fs.exit_code, 1, "Missing --fs argument exits with error code 1");

        auto res_m_dur = run_cli("record --duration");
        ASSERT_EQ(res_m_dur.exit_code, 1, "Missing --duration argument exits with error code 1");

        auto res_m_g = run_cli("record --gain");
        ASSERT_EQ(res_m_g.exit_code, 1, "Missing --gain argument exits with error code 1");

        auto res_m_out = run_cli("record -o");
        ASSERT_EQ(res_m_out.exit_code, 1, "Missing -o argument exits with error code 1");

        auto res_bad_fc = run_cli("record --fc 0");
        ASSERT_EQ(res_bad_fc.exit_code, 1, "Invalid zero center frequency rejected with exit code 1");

        auto res_bad_fs = run_cli("record --fs 0");
        ASSERT_EQ(res_bad_fs.exit_code, 1, "Invalid zero sample rate rejected with exit code 1");
    }

    // -----------------------------------------------------------------------
    // TEST 3: Analyze Subcommand Argument Validation
    // -----------------------------------------------------------------------
    std::cout << "\n[Test 3] Testing Analyze Subcommand Flag Validation...\n";
    {
        auto res_bad_flag = run_cli("analyze --invalid-flag");
        ASSERT_EQ(res_bad_flag.exit_code, 1, "Unrecognized analyze option exits with error code 1");
        ASSERT_TRUE(res_bad_flag.output.find("Unrecognized option for analyze") != std::string::npos,
                    "Reports unrecognized option for analyze");

        auto res_missing_file = run_cli("analyze nonexistent_obs_file_987654.cs16");
        ASSERT_EQ(res_missing_file.exit_code, 1, "Non-existent file exits with code 1");
        ASSERT_TRUE(res_missing_file.output.find("does not exist") != std::string::npos,
                    "Reports file does not exist error");

        auto res_m_f = run_cli("analyze -f");
        ASSERT_EQ(res_m_f.exit_code, 1, "Missing -f argument exits with code 1");

        auto res_m_n = run_cli("analyze -n");
        ASSERT_EQ(res_m_n.exit_code, 1, "Missing -n argument exits with code 1");

        auto res_m_base = run_cli("analyze --baseline");
        ASSERT_EQ(res_m_base.exit_code, 1, "Missing --baseline argument exits with code 1");

        auto res_m_tsys = run_cli("analyze --tsys");
        ASSERT_EQ(res_m_tsys.exit_code, 1, "Missing --tsys argument exits with code 1");

        auto res_m_out = run_cli("analyze -o");
        ASSERT_EQ(res_m_out.exit_code, 1, "Missing -o argument exits with code 1");

        // Create a real temp file so argument validation checks are non-vacuous
        std::string dummy_cs16 = test_dir + "/flag_check.cs16";
        {
            std::ofstream ofs(dummy_cs16, std::ios::binary);
            std::vector<char> zeros(64, 0);
            ofs.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
        }

        auto res_bad_n = run_cli("analyze " + dummy_cs16 + " -n 63");
        ASSERT_EQ(res_bad_n.exit_code, 1, "Odd/too-small FFT size (-n 63) exits with code 1");
        ASSERT_TRUE(res_bad_n.output.find("--fft-size") != std::string::npos,
                    "Reports specific --fft-size error for -n 63 (non-vacuous)");

        auto res_nan_n = run_cli("analyze " + dummy_cs16 + " -n abc");
        ASSERT_EQ(res_nan_n.exit_code, 1, "Non-numeric FFT size (-n abc) exits with code 1");
        ASSERT_TRUE(res_nan_n.output.find("--fft-size") != std::string::npos,
                    "Reports --fft-size error for -n abc");

        auto res_bad_tsys = run_cli("analyze " + dummy_cs16 + " --tsys -5");
        ASSERT_EQ(res_bad_tsys.exit_code, 1, "Non-positive Tsys (--tsys -5) exits with code 1");
        ASSERT_TRUE(res_bad_tsys.output.find("--tsys") != std::string::npos,
                    "Reports specific --tsys error for --tsys -5 (non-vacuous)");

        auto res_nan_tsys = run_cli("analyze " + dummy_cs16 + " --tsys NaN");
        ASSERT_EQ(res_nan_tsys.exit_code, 1, "NaN Tsys (--tsys NaN) exits with code 1");
        ASSERT_TRUE(res_nan_tsys.output.find("--tsys") != std::string::npos,
                    "Reports --tsys error for --tsys NaN");

        auto res_bad_base = run_cli("analyze " + dummy_cs16 + " --baseline bogus");
        ASSERT_EQ(res_bad_base.exit_code, 1, "Unrecognized --baseline bogus exits with code 1");
        ASSERT_TRUE(res_bad_base.output.find("--baseline") != std::string::npos,
                    "Reports unrecognized --baseline error");

        auto res_bad_lm = run_cli("analyze " + dummy_cs16 + " --line-method bogus");
        ASSERT_EQ(res_bad_lm.exit_code, 1, "Unrecognized --line-method bogus exits with code 1");
        ASSERT_TRUE(res_bad_lm.output.find("--line-method") != std::string::npos,
                    "Reports unrecognized --line-method error");

        auto res_bad_win = run_cli("analyze " + dummy_cs16 + " --window bogus");
        ASSERT_EQ(res_bad_win.exit_code, 1, "Unrecognized --window bogus exits with code 1");

        auto res_bad_fc = run_cli("analyze " + dummy_cs16 + " --fc bogus");
        ASSERT_EQ(res_bad_fc.exit_code, 1, "Invalid --fc bogus exits with code 1");

        auto res_bad_fs = run_cli("analyze " + dummy_cs16 + " --fs bogus");
        ASSERT_EQ(res_bad_fs.exit_code, 1, "Invalid --fs bogus exits with code 1");

        auto res_bad_edge = run_cli("analyze " + dummy_cs16 + " --edge-fraction 0.9");
        ASSERT_EQ(res_bad_edge.exit_code, 1, "Out-of-range --edge-fraction 0.9 exits with code 1");
    }

    // -----------------------------------------------------------------------
    // TEST 4: Unknown Subcommand Reporting
    // -----------------------------------------------------------------------
    std::cout << "\n[Test 4] Testing Unknown Subcommand Error Handling...\n";
    {
        auto res_bad_cmd = run_cli("invalid_command_name");
        ASSERT_EQ(res_bad_cmd.exit_code, 1, "Unknown command exits with error code 1");
        ASSERT_TRUE(res_bad_cmd.output.find("Unknown subcommand or file") != std::string::npos,
                    "Reports unknown subcommand error");
    }

    // -----------------------------------------------------------------------
    // TEST 5: Functional Virtual Record & Analyze Execution
    // -----------------------------------------------------------------------
    std::cout << "\n[Test 5] Functional Execution: Virtual SDR Record (1.0s) -> CS16 -> Analyze...\n";
    std::string rec_target = test_dir + "/cli_test.cs16";
    {
        std::string rec_cmd = "record --virtual --fc 1420.5 --fs 2.4 --duration 1.0 -o " + rec_target;
        auto rec_res = run_cli(rec_cmd);
        ASSERT_EQ(rec_res.exit_code, 0, "CLI record subcommand with suffixless --fc 1420.5 --fs 2.4 succeeded");
        ASSERT_TRUE(rec_res.output.find("Recording completed successfully") != std::string::npos,
                    "Terminal reports recording completion");

        ASSERT_TRUE(std::filesystem::exists(rec_target), "Output .cs16 binary generated");
        std::string json_sidecar = test_dir + "/cli_test.json";
        std::string txt_sidecar  = test_dir + "/cli_test.txt";
        ASSERT_TRUE(std::filesystem::exists(json_sidecar), "Output .json sidecar generated");
        ASSERT_TRUE(std::filesystem::exists(txt_sidecar), "Output .txt sidecar generated");

        std::ifstream jifs(json_sidecar);
        std::string jtxt((std::istreambuf_iterator<char>(jifs)), std::istreambuf_iterator<char>());
        ASSERT_TRUE(jtxt.find("2400000") != std::string::npos, "Suffixless --fs 2.4 parsed as 2400000 Hz (2.4 MSPS)");

        // Run analyze on the generated file
        std::string ana_cmd = "analyze " + rec_target + " -o " + out_dir;
        auto ana_res = run_cli(ana_cmd);
        ASSERT_EQ(ana_res.exit_code, 0, "CLI analyze subcommand succeeded with exit code 0");
        ASSERT_TRUE(ana_res.output.find("Peak Frequency:") != std::string::npos, "Reports peak frequency");
        ASSERT_TRUE(ana_res.output.find("Radial Velocity:") != std::string::npos, "Reports radial velocity");
        ASSERT_TRUE(ana_res.output.find("H I Column Density:") != std::string::npos, "Reports H I column density");

        // Verify artifacts exist in out_dir
        ASSERT_TRUE(std::filesystem::exists(out_dir + "/1_hi_spectrum_calibrated.png"), "Artifact 1 exists");
        ASSERT_TRUE(std::filesystem::exists(out_dir + "/5_hi_master_dashboard.png"), "Artifact 5 exists");
        ASSERT_TRUE(std::filesystem::exists(out_dir + "/calibrated_spectrum.csv"), "CSV artifact exists");
        ASSERT_TRUE(std::filesystem::exists(out_dir + "/analysis_report.json"), "JSON report artifact exists");

        // Positive test: advanced baseline/line/window CLI options (--baseline standing_wave --line-method double_gaussian --hi-window-khz 180)
        std::string out_dir_adv = "tests/test_out_cli_adv";
        std::filesystem::remove_all(out_dir_adv, ec);
        std::string adv_cmd = "analyze \"" + rec_target +
                              "\" --baseline standing_wave --line-method double_gaussian --hi-window-khz 180 --window hann -d 1.0 --output \"" +
                              out_dir_adv + "\" --quiet";
        auto adv_res = run_cli(adv_cmd);
        ASSERT_EQ(adv_res.exit_code, 0, "CLI analyze with --baseline standing_wave --line-method double_gaussian --hi-window-khz 180 succeeded");
        std::string adv_json_path = out_dir_adv + "/analysis_report.json";
        ASSERT_TRUE(std::filesystem::exists(adv_json_path), "Advanced CLI analysis emitted analysis_report.json");
        {
            std::ifstream ajifs(adv_json_path);
            std::string ajtxt((std::istreambuf_iterator<char>(ajifs)), std::istreambuf_iterator<char>());
            ASSERT_TRUE(ajtxt.find("\"standing_wave\"") != std::string::npos, "JSON records standing_wave baseline");
            ASSERT_TRUE(ajtxt.find("\"double_gaussian\"") != std::string::npos, "JSON records double_gaussian line method");
            ASSERT_TRUE(ajtxt.find("\"is_detected\"") != std::string::npos &&
                        ajtxt.find("\"snr_matched\"") != std::string::npos,
                        "JSON exports is_detected and snr_matched telemetry");
        }
        std::filesystem::remove_all(out_dir_adv, ec);
    }

    // -----------------------------------------------------------------------
    // TEST 6: Auto-Detection Mode Validation
    // -----------------------------------------------------------------------
    std::cout << "\n[Test 6] Testing Subcommand Auto-Detection Modes...\n";
    {
        std::string auto_out = "tests/test_auto_out";
        std::string auto_cmd = rec_target + " -o " + auto_out;
        auto res_auto_ana = run_cli(auto_cmd);
        ASSERT_EQ(res_auto_ana.exit_code, 0, "Auto-detected analyze mode from .cs16 argument");
        ASSERT_TRUE(res_auto_ana.output.find("Peak Frequency:") != std::string::npos,
                    "Auto-detect run output contains astrophysical metrics");
        std::filesystem::remove_all(auto_out, ec);

        std::string auto_rec_file = test_dir + "/auto_rec.cs16";
        std::string auto_rec_cmd = "--virtual --gain 25.4 --duration 1.0 -o " + auto_rec_file;
        auto res_auto_rec = run_cli(auto_rec_cmd);
        ASSERT_EQ(res_auto_rec.exit_code, 0, "Auto-detected record mode from record flags");
        ASSERT_TRUE(std::filesystem::exists(auto_rec_file), "Auto-detected record created binary");
    }

    // Cleanup
    std::filesystem::remove_all(test_dir, ec);
    std::filesystem::remove_all(out_dir, ec);

    std::cout << "\n----------------------------------------------------------------------\n";
    std::cout << "RESULTS: " << g_tests_passed << " / " << g_tests_run << " assertions PASSED.\n";
    if (g_tests_passed == g_tests_run) {
        std::cout << "[SUCCESS] test_cli_commands PASSED (100%)\n";
        return 0;
    } else {
        std::cerr << "[FAILURE] " << (g_tests_run - g_tests_passed) << " assertions FAILED!\n";
        return 1;
    }
}
