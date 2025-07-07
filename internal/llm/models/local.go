package models

import (
	"bytes"
	"cmp"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"unicode"

	"github.com/opencode-ai/opencode/internal/logging"
	"github.com/spf13/viper"
)

const (
	ProviderLocal ModelProvider = "local"

	localModelsPath        = "v1/models"
	lmStudioBetaModelsPath = "api/v0/models"
	slotsPath              = "/slots"
)

const (
	// Define a specific log file path for mypackage's init logs if needed,
	// or use the application's main log file path if known and desired.
	// For this specific constraint, let's assume it can be the same file path.
	myPackageLogFilePath  = "./.opencode/init.log"
	myPackageInitLogLevel = slog.LevelDebug // Level for logs from mypackage's init()
)

var (
	myPackageInitFile   *os.File     // Global to mypackage to hold the opened file handle
	myPackageInitLogger *slog.Logger // Local logger for mypackage.init()
	initLoggerSetupOnce sync.Once    // Ensures init setup only runs once
)

func init() {
	initLoggerSetupOnce.Do(func() {
		// Attempt to open the log file for mypackage's init logs.
		// This will ensure logs from this specific init() go to the file.
		logDir := filepath.Dir(myPackageLogFilePath)
		if err := os.MkdirAll(logDir, 0755); err != nil {
			// Fallback to stderr if we can't even create the directory
			myPackageInitLogger = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
				AddSource: true,
				Level:     slog.LevelError,
			}))
			myPackageInitLogger.Error("CRITICAL: Failed to create log directory for mypackage init logs", "error", err, "path", logDir)
			return // Cannot proceed with file logging for init
		}

		file, err := os.OpenFile(myPackageLogFilePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			// Fallback to stderr if we can't open the file
			myPackageInitLogger = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
				AddSource: true,
				Level:     slog.LevelError,
			}))
			myPackageInitLogger.Error("CRITICAL: Failed to open log file for mypackage init logs", "error", err, "path", myPackageLogFilePath)
			return // Cannot proceed with file logging for init
		}
		myPackageInitFile = file // Store the file handle

		// Create a specific logger for mypackage's init function.
		myPackageInitLogger = slog.New(slog.NewTextHandler(myPackageInitFile, &slog.HandlerOptions{
			AddSource: true,
			Level:     myPackageInitLogLevel,
		}))

		myPackageInitLogger.Info("MyPackage: Init file logger set up.")
	})

	myPackageInitLogger.Debug("called local init")

	if endpoint := os.Getenv("LOCAL_ENDPOINT"); endpoint != "" {
		localEndpoint, err := url.Parse(endpoint)
		if err != nil {
			logging.Debug("Failed to parse local endpoint",
				"error", err,
				"endpoint", endpoint,
			)
			return
		}

		load := func(url *url.URL, path string) []localModel {
			url.Path = path
			return listLocalModels(url.String())
		}

		models := load(localEndpoint, lmStudioBetaModelsPath)

		if len(models) == 0 {
			models = load(localEndpoint, localModelsPath)
		}
		myPackageInitLogger.Debug("loaded models", "models", models)
		if len(models) == 0 {
			myPackageInitLogger.Debug("No local models found",
				"endpoint", endpoint,
			)
			return
		}

		loadSlots := func(url *url.URL, path string) []localSlot {
			url.Path = path
			return listLocalSlots(url.String())
		}
		slots := loadSlots(localEndpoint, slotsPath)
		for is, slot := range slots {
			for im, m := range models {
				myPackageInitLogger.Debug("setting ctx form slot", "NCtx", slot.NCtx)
				if im == is {
					models[im] = localModel{
						ID:                  m.ID,
						Object:              m.Object,
						Type:                m.Type,
						Publisher:           m.Publisher,
						Arch:                m.Arch,
						CompatibilityType:   m.CompatibilityType,
						Quantization:        m.Quantization,
						State:               m.State,
						MaxContextLength:    slot.NCtx,
						LoadedContextLength: slot.NCtx,
					}
				}
			}
		}
		loadLocalModels(models)

		viper.SetDefault("providers.local.apiKey", "dummy")
		ProviderPopularity[ProviderLocal] = 0
	}
}

type localModelList struct {
	Data []localModel `json:"data"`
}

type localModel struct {
	ID                  string `json:"id"`
	Object              string `json:"object"`
	Type                string `json:"type"`
	Publisher           string `json:"publisher"`
	Arch                string `json:"arch"`
	CompatibilityType   string `json:"compatibility_type"`
	Quantization        string `json:"quantization"`
	State               string `json:"state"`
	MaxContextLength    int64  `json:"max_context_length"`
	LoadedContextLength int64  `json:"loaded_context_length"`
}

type localSlotList []localSlot

type localSlot struct {
	ID           int   `json:"id"`
	IDTask       int   `json:"id_task"`
	NCtx         int64 `json:"n_ctx"`
	Speculative  bool  `json:"speculative"`
	IsProcessing bool  `json:"is_processing"`
	Params       struct {
		NPredict            int      `json:"n_predict"`
		Seed                uint32   `json:"seed"`
		Temperature         float64  `json:"temperature"`
		DynatempRange       float64  `json:"dynatemp_range"`
		DynatempExponent    float64  `json:"dynatemp_exponent"`
		TopK                int      `json:"top_k"`
		TopP                float64  `json:"top_p"`
		MinP                float64  `json:"min_p"`
		TopNSigma           float64  `json:"top_n_sigma"`
		XtcProbability      float64  `json:"xtc_probability"`
		XtcThreshold        float64  `json:"xtc_threshold"`
		TypicalP            float64  `json:"typical_p"`
		RepeatLastN         int      `json:"repeat_last_n"`
		RepeatPenalty       float64  `json:"repeat_penalty"`
		PresencePenalty     float64  `json:"presence_penalty"`
		FrequencyPenalty    float64  `json:"frequency_penalty"`
		DryMultiplier       float64  `json:"dry_multiplier"`
		DryBase             float64  `json:"dry_base"`
		DryAllowedLength    int      `json:"dry_allowed_length"`
		DryPenaltyLastN     int      `json:"dry_penalty_last_n"`
		DrySequenceBreakers []string `json:"dry_sequence_breakers"`
		Mirostat            int      `json:"mirostat"`
		MirostatTau         float64  `json:"mirostat_tau"`
		MirostatEta         float64  `json:"mirostat_eta"`
		Stop                []any    `json:"stop"` // Can be string or other types based on actual usage
		MaxTokens           int      `json:"max_tokens"`
		NKeep               int      `json:"n_keep"`
		NDiscard            int      `json:"n_discard"`
		IgnoreEos           bool     `json:"ignore_eos"`
		Stream              bool     `json:"stream"`
		LogitBias           []any    `json:"logit_bias"` // Can be array of arrays, etc.
		NProbs              int      `json:"n_probs"`
		MinKeep             int      `json:"min_keep"`
		Grammar             string   `json:"grammar"`
		GrammarLazy         bool     `json:"grammar_lazy"`
		GrammarTriggers     []any    `json:"grammar_triggers"` // Can be array of strings, etc.
		PreservedTokens     []any    `json:"preserved_tokens"`
		ChatFormat          string   `json:"chat_format"`
		ReasoningFormat     string   `json:"reasoning_format"`
		ReasoningInContent  bool     `json:"reasoning_in_content"`
		ThinkingForcedOpen  bool     `json:"thinking_forced_open"`
		Samplers            []string `json:"samplers"`
		SpeculativeNMax     int      `json:"speculative.n_max"`
		SpeculativeNMin     int      `json:"speculative.n_min"`
		SpeculativePMin     float64  `json:"speculative.p_min"`
		TimingsPerToken     bool     `json:"timings_per_token"`
		PostSamplingProbs   bool     `json:"post_sampling_probs"`
		Lora                []any    `json:"lora"` // Can be array of objects
	} `json:"params"`
	Prompt    string `json:"prompt"`
	NextToken struct {
		HasNextToken bool   `json:"has_next_token"`
		HasNewLine   bool   `json:"has_new_line"`
		NRemain      int    `json:"n_remain"`
		NDecoded     int    `json:"n_decoded"`
		StoppingWord string `json:"stopping_word"`
	} `json:"next_token"`
}

func listLocalModels(modelsEndpoint string) []localModel {
	myPackageInitLogger.Debug("requesting models from", "endpoint", modelsEndpoint)
	res, err := http.Get(modelsEndpoint)
	bodyBytes, err := io.ReadAll(res.Body)
	myPackageInitLogger.Debug("modelList from server", "list", string(bodyBytes))
	if err != nil {
		myPackageInitLogger.Debug("Failed to list local models",
			"error", err,
			"endpoint", modelsEndpoint,
		)
		return []localModel{}
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		myPackageInitLogger.Debug("Failed to list local models",
			"status", res.StatusCode,
			"endpoint", modelsEndpoint,
		)
		return []localModel{}
	}

	var modelList localModelList
	if err = json.NewDecoder(bytes.NewReader(bodyBytes)).Decode(&modelList); err != nil {
		myPackageInitLogger.Debug("Failed to list local models",
			"error", err,
			"endpoint", modelsEndpoint,
		)
		return []localModel{}
	}

	var supportedModels []localModel
	for _, model := range modelList.Data {
		if strings.HasSuffix(modelsEndpoint, lmStudioBetaModelsPath) {
			if model.Object != "model" || model.Type != "llm" {
				myPackageInitLogger.Debug("Skipping unsupported LMStudio model",
					"endpoint", modelsEndpoint,
					"id", model.ID,
					"object", model.Object,
					"type", model.Type,
				)

				continue
			}
		}

		supportedModels = append(supportedModels, model)
	}

	return supportedModels
}

func listLocalSlots(slotEndPoint string) []localSlot {
	myPackageInitLogger.Debug("requesting slots from", "endpoint", slotEndPoint)
	res, err := http.Get(slotEndPoint)
	if err != nil {
		myPackageInitLogger.Debug("Failed to list local slots",
			"error", err,
			"endpoint", slotEndPoint,
		)
		return []localSlot{}
	}
	defer res.Body.Close()
	bodyBytes, err := io.ReadAll(res.Body)
	if res.StatusCode != http.StatusOK {
		myPackageInitLogger.Debug("Failed to list local slots",
			"status", res.StatusCode,
			"endpoint", slotEndPoint,
		)
		return []localSlot{}
	}
	var slotList localSlotList
	if err = json.NewDecoder(bytes.NewReader(bodyBytes)).Decode(&slotList); err != nil {
		myPackageInitLogger.Debug("Failed to list local slots",
			"error", err,
			"endpoint", slotEndPoint,
		)
		return []localSlot{}
	}

	myPackageInitLogger.Debug("got slots", "localSlots", slotList)
	return slotList
}

func loadLocalModels(models []localModel) {
	for i, m := range models {
		model := convertLocalModel(m)
		SupportedModels[model.ID] = model

		if i == 0 || m.State == "loaded" {
			viper.SetDefault("agents.coder.model", model.ID)
			viper.SetDefault("agents.summarizer.model", model.ID)
			viper.SetDefault("agents.task.model", model.ID)
			viper.SetDefault("agents.title.model", model.ID)
		}
	}
}

func convertLocalModel(model localModel) Model {
	return Model{
		ID:                  ModelID("local." + model.ID),
		Name:                friendlyModelName(model.ID),
		Provider:            ProviderLocal,
		APIModel:            model.ID,
		ContextWindow:       cmp.Or(model.LoadedContextLength, 4096),
		DefaultMaxTokens:    cmp.Or(model.LoadedContextLength, 4096),
		CanReason:           true,
		SupportsAttachments: true,
	}
}

var modelInfoRegex = regexp.MustCompile(`(?i)^([a-z0-9]+)(?:[-_]?([rv]?\d[\.\d]*))?(?:[-_]?([a-z]+))?.*`)

func friendlyModelName(modelID string) string {
	mainID := modelID
	tag := ""

	if slash := strings.LastIndex(mainID, "/"); slash != -1 {
		mainID = mainID[slash+1:]
	}

	if at := strings.Index(modelID, "@"); at != -1 {
		mainID = modelID[:at]
		tag = modelID[at+1:]
	}

	match := modelInfoRegex.FindStringSubmatch(mainID)
	if match == nil {
		return modelID
	}

	capitalize := func(s string) string {
		if s == "" {
			return ""
		}
		runes := []rune(s)
		runes[0] = unicode.ToUpper(runes[0])
		return string(runes)
	}

	family := capitalize(match[1])
	version := ""
	label := ""

	if len(match) > 2 && match[2] != "" {
		version = strings.ToUpper(match[2])
	}

	if len(match) > 3 && match[3] != "" {
		label = capitalize(match[3])
	}

	var parts []string
	if family != "" {
		parts = append(parts, family)
	}
	if version != "" {
		parts = append(parts, version)
	}
	if label != "" {
		parts = append(parts, label)
	}
	if tag != "" {
		parts = append(parts, tag)
	}

	return strings.Join(parts, " ")
}
