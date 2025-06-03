# trading_widget_config.py
# SibelGPT TradingView Widget Konfigürasyonları

from typing import Dict, List, Any

class TradingWidgetConfig:
    """TradingView widget konfigürasyonlarını yönetir"""
    
    # Ana Türk Piyasa Sembolleri
    MAIN_TURKISH_SYMBOLS = [
        {
            "description": "BIST100",
            "proName": "BIST:XU100"
        },
        {
            "description": "USDTRY", 
            "proName": "FX:USDTRY"
        },
        {
            "description": "GRAM ALTIN",
            "proName": "FX_IDC:XAUTRYG"
        },
        {
            "description": "BITCOIN",
            "proName": "BINANCE:BTCUSD"
        },
        {
            "description": "EUR/USD",
            "proName": "FOREXCOM:EURUSD"
        }
    ]
    
    # Tema Ayarları
    WIDGET_THEMES = {
        "dark": "dark",
        "light": "light"
    }
    
    # Widget Temel Ayarları
    DEFAULT_TICKER_CONFIG = {
        "isTransparent": False,
        "showSymbolLogo": True, 
        "colorTheme": "dark",
        "locale": "tr"
    }
    
    @staticmethod
    def get_main_tickers_config() -> Dict[str, Any]:
        """Ana ticker widget konfigürasyonunu döndürür"""
        return {
            "symbols": TradingWidgetConfig.MAIN_TURKISH_SYMBOLS,
            **TradingWidgetConfig.DEFAULT_TICKER_CONFIG
        }
    
    @staticmethod
    def generate_ticker_html(config: Dict[str, Any] = None) -> str:
        """Ticker widget HTML kodunu oluşturur"""
        if config is None:
            config = TradingWidgetConfig.get_main_tickers_config()
            
        import json
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        
        html = f'''
        <!-- TradingView Ticker Widget -->
        <div class="tradingview-widget-container" id="tv-ticker-widget">
            <div class="tradingview-widget-container__widget"></div>
            <div class="tradingview-widget-copyright">
                <a href="https://tr.tradingview.com/" rel="noopener nofollow" target="_blank">
                    <span class="blue-text">Tüm piyasaları TradingView üzerinden takip edin</span>
                </a>
            </div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-tickers.js" async>
            {config_json}
            </script>
        </div>
        <!-- TradingView Ticker Widget END -->
        '''
        return html

# Test fonksiyonu
if __name__ == "__main__":
    config = TradingWidgetConfig()
    print("✅ Widget Config Test:")
    print(config.get_main_tickers_config())
    print("\n✅ HTML Test:")
    print(config.generate_ticker_html())
