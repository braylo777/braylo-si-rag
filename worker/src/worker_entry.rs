//! Cloudflare Worker entry (wasm only). Wires the bindings (AI, VECTORS, TRACES, RATE_LIMITER) into the `Host` trait.
use crate::agent::Host;
use crate::api::handle;
use crate::obs::Kv;
use crate::page::PAGE;
use js_sys::{Array, Function, Object, Promise, Reflect};
use serde_json::{json, Value};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use worker::*;

struct Bindings {
    env: Env,
    ip: String,
}

fn binding(env: &Env, name: &str) -> Option<JsValue> {
    let v = Reflect::get(env.as_ref(), &name.into()).ok()?;
    if v.is_undefined() || v.is_null() {
        None
    } else {
        Some(v)
    }
}

async fn call(target: &JsValue, method: &str, args: &[JsValue]) -> std::result::Result<JsValue, String> {
    let f: Function = Reflect::get(target, &method.into()).map_err(|e| format!("{e:?}"))?.dyn_into().map_err(|_| format!("{method} is not a function"))?;
    let a = Array::new();
    for x in args {
        a.push(x);
    }
    let r = f.apply(target, &a).map_err(|e| format!("{e:?}"))?;
    match r.dyn_into::<Promise>() {
        Ok(p) => JsFuture::from(p).await.map_err(|e| format!("{e:?}")),
        Err(v) => Ok(v),
    }
}

impl Kv for Bindings {
    async fn get(&self, key: &str) -> Option<String> {
        let kv = self.env.kv("TRACES").ok()?;
        kv.get(key).text().await.ok().flatten()
    }
    async fn put(&self, key: &str, val: &str) {
        if let Ok(kv) = self.env.kv("TRACES") {
            if let Ok(b) = kv.put(key, val) {
                let _ = b.execute().await;
            }
        }
    }
    async fn delete(&self, key: &str) {
        if let Ok(kv) = self.env.kv("TRACES") {
            let _ = kv.delete(key).await;
        }
    }
}

impl Host for Bindings {
    fn has_ai(&self) -> bool {
        binding(&self.env, "AI").is_some()
    }
    fn has_vectors(&self) -> bool {
        binding(&self.env, "VECTORS").is_some()
    }
    fn has_kv(&self) -> bool {
        binding(&self.env, "TRACES").is_some()
    }
    async fn ai(&self, model: &str, input: Value) -> std::result::Result<Value, String> {
        let ai = self.env.ai("AI").map_err(|e| e.to_string())?;
        ai.run::<Value, Value>(model, input).await.map_err(|e| e.to_string())
    }
    async fn vectors_query(&self, vec: Vec<f64>, top_k: usize) -> std::result::Result<Vec<String>, String> {
        let v = binding(&self.env, "VECTORS").ok_or("no vector index")?;
        let arr = Array::new();
        for x in vec {
            arr.push(&JsValue::from_f64(x));
        }
        let opts = Object::new();
        Reflect::set(&opts, &"topK".into(), &JsValue::from_f64(top_k as f64)).map_err(|e| format!("{e:?}"))?;
        let res = call(&v, "query", &[arr.into(), opts.into()]).await?;
        let matches = Reflect::get(&res, &"matches".into()).map_err(|e| format!("{e:?}"))?;
        let mut out = vec![];
        for m in Array::from(&matches).iter() {
            if let Some(id) = Reflect::get(&m, &"id".into()).ok().and_then(|i| i.as_string()) {
                out.push(id);
            }
        }
        Ok(out)
    }
    async fn limited(&self) -> bool {
        let Some(rl) = binding(&self.env, "RATE_LIMITER") else { return false };
        let o = Object::new();
        let _ = Reflect::set(&o, &"key".into(), &self.ip.clone().into());
        match call(&rl, "limit", &[o.into()]).await {
            Ok(r) => !Reflect::get(&r, &"success".into()).ok().and_then(|s| s.as_bool()).unwrap_or(true),
            Err(_) => false,
        }
    }
    fn now(&self) -> f64 {
        js_sys::Date::now()
    }
    fn epoch(&self) -> i64 {
        js_sys::Date::now() as i64
    }
    fn trace_id(&self) -> String {
        let c = Reflect::get(&js_sys::global(), &"crypto".into()).unwrap_or(JsValue::UNDEFINED);
        let id = Reflect::get(&c, &"randomUUID".into())
            .ok()
            .and_then(|f| f.dyn_into::<Function>().ok())
            .and_then(|f| f.call0(&c).ok())
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        id.replace('-', "")
    }
}

#[event(fetch)]
async fn fetch(mut req: Request, env: Env, _ctx: Context) -> Result<Response> {
    let url = req.url()?;
    let ip = req.headers().get("cf-connecting-ip")?.unwrap_or_else(|| "anon".into());
    let method = req.method().to_string();
    let body: Value = if method == "POST" { req.json().await.unwrap_or_else(|_| json!({})) } else { json!({}) };
    let host = Bindings { env, ip };
    let (status, value, extra) = handle(&host, &method, url.path(), url.query().unwrap_or(""), &body).await;
    if status == 0 {
        let mut r = Response::from_html(PAGE)?;
        r.headers_mut().set("content-type", "text/html; charset=utf-8")?;
        return Ok(r);
    }
    let mut r = Response::from_json(&value)?.with_status(status);
    for (k, v) in extra {
        r.headers_mut().set(k, &v)?;
    }
    Ok(r)
}
