use gst::glib;
use gstreamer as gst;

mod embed;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    embed::register(plugin)?;
    Ok(())
}

gst::plugin_define!(
    aguawatermark,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    env!("CARGO_PKG_VERSION"),
    "MIT OR Apache-2.0",
    env!("CARGO_PKG_NAME"),
    "agua",
    env!("CARGO_PKG_REPOSITORY"),
    "2026-02-23"
);
