"""Methods modül - Sis giderme yöntemleri."""

from methods.dcp import DCPMethod
from methods.clahe import CLAHEMethod
from methods.retinex import RetinexMethod

# Factory registry: yeni yöntem eklemek için sadece buraya kayıt yeterli (Open/Closed)
METHOD_REGISTRY: dict[str, type] = {
    "dcp": DCPMethod,
    "clahe": CLAHEMethod,
    "retinex": RetinexMethod,
}


def create_method(name: str, **kwargs):
    """
    Factory fonksiyonu: isimle yöntem oluşturur.

    Args:
        name: Yöntem adı ("dcp", "clahe", "retinex").
        **kwargs: Yönteme özgü parametreler.

    Returns:
        IDehazingMethod implementasyonu.

    Raises:
        ValueError: Bilinmeyen yöntem adı.
    """
    name = name.lower().strip()
    if name not in METHOD_REGISTRY:
        available = ", ".join(METHOD_REGISTRY.keys())
        raise ValueError(f"Bilinmeyen yöntem: '{name}'. Mevcut yöntemler: {available}")
    return METHOD_REGISTRY[name](**kwargs)


def get_all_methods(**kwargs) -> list:
    """Tüm kayıtlı yöntemlerden birer örnek oluşturur."""
    return [cls(**kwargs) for cls in METHOD_REGISTRY.values()]
