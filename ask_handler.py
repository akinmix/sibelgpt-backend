def format_context_for_sibelgpt(listings: List[Dict]) -> str:
    if not listings:
        return "🔍 Uygun ilan bulunamadı."

    # Türkçe locale (opsiyonel)
    try:
        locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'tr_TR')
        except locale.Error:
            pass

    formatted_parts = []
    for i, l in enumerate(listings, start=1):
        ilan_no    = l.get("ilan_no", "(numara yok)")
        baslik     = l.get("baslik", "(başlık yok)")
        lokasyon   = l.get("lokasyon", "?")
        fiyat_raw  = l.get("fiyat")
        ozellikler = l.get("ozellikler", "(özellik yok)")
        fiyat      = "?"

        # Fiyatı locale ile formatla
        try:
            fiyat_num = float(str(fiyat_raw).replace('.', '').replace(',', '.'))
            try:
                fiyat = locale.currency(fiyat_num, symbol='₺', grouping=True)
                if fiyat.endswith('.00') or fiyat.endswith(',00'):
                    fiyat = fiyat[:-3] + ' ₺'
                else:
                    fiyat = fiyat.replace('₺', '').strip() + ' ₺'
            except:
                fiyat = f"{fiyat_num:,.0f} ₺".replace(',', '#').replace('.', ',').replace('#', '.')
        except:
            fiyat = str(fiyat_raw) if fiyat_raw else "?"

        # HTML <ul><li> formatı
        ilan_html = (
            f"<li>"
            f"<strong>{i}. {baslik}</strong><br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• İlan No: {ilan_no}<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• Lokasyon: {lokasyon}<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• Fiyat: {fiyat}<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• Özellikler: {ozellikler}"
            f"</li><br>"
        )
        formatted_parts.append(ilan_html)

    final_output = "<ul>" + "".join(formatted_parts) + "</ul>"
    final_output += "<br>📞 Bu ilanlar hakkında daha fazla bilgi almak isterseniz: 532 687 84 64"

    return final_output
