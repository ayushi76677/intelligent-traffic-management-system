/**
 * @googlemaps/js-api-loader - Browser ES Module
 * Official Google Maps Platform dynamic bootstrap loader
 * Exports: setOptions, importLibrary, Loader
 */

const TRUSTED_TYPES_POLICY_NAME = "@googlemaps/js-api-loader";
const fallbackPolicy = { createScriptURL: (url) => url };
let policy;

function getPolicy() {
    if (policy) return policy;
    const trustedTypes = globalThis.trustedTypes;
    if (!trustedTypes) {
        policy = fallbackPolicy;
        return policy;
    }
    try {
        policy = trustedTypes.createPolicy(TRUSTED_TYPES_POLICY_NAME, {
            createScriptURL: (url) => url,
        });
    } catch (e) {
        policy = fallbackPolicy;
    }
    return policy;
}

function setScriptSrc(script, src) {
    script.src = getPolicy().createScriptURL(src);
}

const bootstrap = (bootstrapParams) => {
    var bootstrapPromise;
    var script;
    var bootstrapParamsKey;
    var PRODUCT_NAME = "The Google Maps JavaScript API";
    var GOOGLE = "google";
    var IMPORT_API_NAME = "importLibrary";
    var PENDING_BOOTSTRAP_KEY = "__ib__";
    var doc = document;
    var global_ = window;
    var google_ = global_[GOOGLE] || (global_[GOOGLE] = {});
    var namespace = google_.maps || (google_.maps = {});
    var libraries = new Set();
    var searchParams = new URLSearchParams();

    var triggerBootstrap = () => bootstrapPromise || (bootstrapPromise = new Promise(async (resolve, reject) => {
        await (script = doc.createElement("script"));
        searchParams.set("libraries", [...libraries] + "");
        for (bootstrapParamsKey in bootstrapParams) {
            searchParams.set(bootstrapParamsKey.replace(/[A-Z]/g, (g) => "_" + g[0].toLowerCase()), bootstrapParams[bootstrapParamsKey]);
        }
        searchParams.set("callback", GOOGLE + ".maps." + PENDING_BOOTSTRAP_KEY);
        setScriptSrc(script, "https://maps.googleapis.com/maps/api/js?" + searchParams);
        namespace[PENDING_BOOTSTRAP_KEY] = resolve;
        script.onerror = () => bootstrapPromise = reject(Error(PRODUCT_NAME + " could not load."));
        script.nonce = doc.querySelector("script[nonce]")?.nonce || "";
        doc.head.append(script);
    }));

    namespace[IMPORT_API_NAME] ? console.warn(PRODUCT_NAME + " only loads once. Ignoring:", bootstrapParams) : namespace[IMPORT_API_NAME] = (libraryName, ...args) => libraries.add(libraryName) && triggerBootstrap().then(() => namespace[IMPORT_API_NAME](libraryName, ...args));
};

let setOptionsWasCalled_ = false;

export function setOptions(options) {
    if (setOptionsWasCalled_) {
        return;
    }
    if (options.apiKey && !options.key) {
        options.key = options.apiKey;
    }
    const importLibraryExists = Boolean(window.google?.maps?.importLibrary);
    if (!importLibraryExists) {
        bootstrap(options);
    }
    setOptionsWasCalled_ = true;
}

export async function importLibrary(libraryName) {
    if (!setOptionsWasCalled_) {
        console.warn("[@googlemaps/js-api-loader] setOptions() was not called before importLibrary().");
    }
    if (!window?.google?.maps?.importLibrary) {
        throw new Error("google.maps.importLibrary is not installed.");
    }
    return (await google.maps.importLibrary(libraryName));
}

export class Loader {
    constructor() {
        throw new Error("The Loader class is deprecated. Use setOptions() and importLibrary().");
    }
}
